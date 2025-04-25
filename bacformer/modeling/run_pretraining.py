import json
import logging
import os
from functools import partial

import numpy as np
from transformers import EarlyStoppingCallback, TrainingArguments, set_seed

from bacformer.modeling.argparser import BacformerArgumentParser
from bacformer.modeling.config import SPECIAL_TOKENS_DICT, BacformerConfig
from bacformer.modeling.data_reader import collate_genome_samples, fetch_training_data
from bacformer.modeling.modeling_pretraining import BacformerForCausalGM, BacformerForMaskedGM
from bacformer.modeling.trainer import BacformerTrainer
from bacformer.modeling.utils import find_ckpt_in_dir, get_gpu_info, pretraining_metrics_fn


def run(
    args: BacformerArgumentParser,
):
    """Train the model."""
    # create config and model
    config = BacformerConfig.from_dict(args.as_dict())

    if args.is_causal_gm and args.mgm_probability > 0.0:
        logging.info(
            "is_causal_gm flag is set to true and mgm_probability is > 0. "
            "Please pick one of the two. Setting mgm_probability to 0.0 and using causal GM."
        )
        args.mgm_probability = 0.0
    elif not args.is_causal_gm and args.mgm_probability == 0.0:
        raise ValueError("Please set either is_causal_gm to True or mgm_probability to a value > 0.0")

    if args.is_causal_gm:
        model = BacformerForCausalGM(config)
    else:
        model = BacformerForMaskedGM(config)

    # log the number of parameters
    logging.info(f"Nr of parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info(f"Nr of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # get datasets
    print("Input dir:", args.input_dir)
    data = fetch_training_data(
        input_dir=args.input_dir,
        mgm_probability=args.mgm_probability,
        test=args.test,
        random_state=args.random_state,
    )

    # get GPU info accounting for training on GPU, XPU and CPU
    n_gpus, use_ipex = get_gpu_info()
    # multiply the number of GPUs per nr of nodes
    n_gpus = n_gpus * args.n_nodes
    # scale the n_steps and nr of warmup steps based on nr of grad_accum_steps and n_gpus
    # assuming using DDP when n_gpus > 1
    n_steps = (
        args.max_epochs * args.n_total_samples // (args.batch_size * args.gradient_accumulation_steps * max(n_gpus, 1))
    )
    warmup_steps = int(n_steps * args.warmup_proportion)
    # scale the learning rare with n of gpus in DDP setup,
    # see https://github.com/Lightning-AI/pytorch-lightning/discussions/3706
    lr = args.lr * np.sqrt(max(n_gpus, 1))

    # tf32 only works on ampere nodes
    use_tf32 = False
    if n_gpus > 1 and not use_ipex:
        use_tf32 = True

    # use gradient checkpointing when we cannot fit a batch size of 1 into a GPU
    use_grad_checkpointing = False
    if n_gpus > 0 and not use_ipex and args.num_hidden_layers > 6:
        print("Using gradient checkpointing for training.")
        use_grad_checkpointing = True

    # get training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps" if args.eval_steps > 0 else "epoch",
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        learning_rate=lr,
        save_strategy="steps" if args.save_steps > 0 else "epoch",
        save_steps=args.save_steps if args.save_steps > 0 else None,
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # add max number of steps to get nr of warmup steps
        max_steps=n_steps,
        # use cpu for local debugging
        use_cpu=False if n_gpus > 0 else True,
        logging_steps=args.logging_steps,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to=["tensorboard"],
        logging_dir=os.path.join(args.output_dir, "logs"),
        seed=args.random_state,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=True if n_gpus > 0 else False,
        weight_decay=args.weight_decay,
        metric_for_best_model=args.monitor_metric,
        load_best_model_at_end=True,
        warmup_steps=warmup_steps,
        greater_is_better=False if "loss" in args.monitor_metric else True,
        prediction_loss_only=True,
        eval_accumulation_steps=1,
        tf32=use_tf32,
        ignore_data_skip=True,
        # fix for DDP training with IterableDataset
        dispatch_batches=False,
        split_batches=False,
        ddp_find_unused_parameters=False if use_grad_checkpointing else True,
        remove_unused_columns=False,
        use_ipex=use_ipex,
        # use gradient checkpointing when we cannot fit a batch size of 1 into a GPU
        gradient_checkpointing=use_grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_broadcast_buffers=False,
    )

    collate_genome_samples_fn = partial(
        collate_genome_samples, SPECIAL_TOKENS_DICT["PAD"], args.max_n_proteins, args.max_n_contigs
    )
    trainer = BacformerTrainer(
        model=model,
        data_collator=collate_genome_samples_fn,
        train_dataset=data.train_dataset,
        eval_dataset=data.val_dataset,
        args=training_args,
        compute_metrics=pretraining_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    # resume training from a checkpoint
    # support automatic restarting
    ckpt_path = find_ckpt_in_dir(args.output_dir)
    trainer.train(resume_from_checkpoint=ckpt_path)


def main(args):
    """Train the model."""
    set_seed(args.random_state)

    # write the arguments for reproducibility
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.as_dict(), f)

    # run training
    run(args)


if __name__ == "__main__":
    args = BacformerArgumentParser().parse_args()
    main(args)
