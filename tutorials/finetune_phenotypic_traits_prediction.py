import os
from functools import partial

import torch
from bacformer.modeling import (
    SPECIAL_TOKENS_DICT,
    BacformerTrainer,
    collate_genome_samples,
    compute_metrics_binary_genome_pred,
)
from bacformer.pp import dataset_col_to_bacformer_inputs
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, EarlyStoppingCallback, TrainingArguments


def run():
    """Train a Bacformer model to predict a phenotypic trait from protein sequences."""
    # 1. Prepare a data sample for a specified phenotypic trait. (look at performance)
    # 2. Load and embed the data
    # 3. Finetune a model to predict the trait
    # 4. Evaluate the model performance
    # 5. Look at the results (do some plot)

    dataset = load_dataset("macwiatrak/phenotypic-trait-catalase-protein-sequences", keep_in_memory=False)

    # embed the protein sequences with the ESM-2 base model
    for split_name in dataset.keys():
        dataset[split_name] = dataset_col_to_bacformer_inputs(
            dataset=dataset[split_name],
            protein_sequences_col="sequence",
            max_n_proteins=7000,
        )

    # load the Bacformer model for genome classification
    # for this task we use the Bacformer model trained on masked complete genomes
    bacformer_model = AutoModelForSequenceClassification.from_pretrained(
        "macwiatrak/bacformer-masked-complete-genomes", trust_remote_code=True
    ).to(torch.bfloat16)
    print("Nr of parameters:", sum(p.numel() for p in bacformer_model.parameters()))
    print("Nr of trainable parameters:", sum(p.numel() for p in bacformer_model.parameters() if p.requires_grad))

    # create a trainer
    # get training args
    output_dir = "output/pheno_trait_pred"
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=0.00015,
        num_train_epochs=50,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        seed=12,
        dataloader_num_workers=4,
        bf16=True,
        metric_for_best_model="eval_auroc",
        load_best_model_at_end=True,
        greater_is_better=True,
    )

    # define a collate function for the dataset
    collate_genome_samples_fn = partial(collate_genome_samples, SPECIAL_TOKENS_DICT["PAD"], 9000, 1000)
    trainer = BacformerTrainer(
        model=bacformer_model,
        data_collator=collate_genome_samples_fn,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        compute_metrics=compute_metrics_binary_genome_pred,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # train the model
    trainer.train()

    # evaluate the model on the validation set
    val_output = trainer.predict(dataset["validation"])
    print("Validation output:", val_output.metrics)
    test_output = trainer.predict(dataset["test"])
    print("Test output:", test_output.metrics)

    # use the trained model to predict the trait on a single new genome using an assembly


if __name__ == "__main__":
    run()
