import os
from functools import partial

import torch
from bacformer.modeling import (
    SPECIAL_TOKENS_DICT,
    BacformerTrainer,
    collate_genome_samples,
    compute_metrics_gene_essentiality_pred,
)
from bacformer.pp import dataset_col_to_bacformer_inputs
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, EarlyStoppingCallback, TrainingArguments


def run():
    """Train the gene essentiality prediction model."""
    # load the dataset
    dataset = load_dataset("macwiatrak/bacbench-essential-genes-protein-sequences")

    # embed the protein sequences with the ESM-2 base model
    for split_name in dataset.keys():
        dataset[split_name] = dataset_col_to_bacformer_inputs(
            dataset=dataset[split_name],
            protein_sequences_col="sequence",
            max_n_proteins=9000,
        )

    # load the Bacformer model for protein classification
    # for this task we use the Bacformer model trained on masked complete genomes
    # with a token (here protein) classification head
    bacformer_model = AutoModelForTokenClassification.from_pretrained(
        "macwiatrak/bacformer-masked-complete-genomes", trust_remote_code=True
    ).to(torch.bfloat16)
    print("Nr of parameters:", sum(p.numel() for p in bacformer_model.parameters()))
    print("Nr of trainable parameters:", sum(p.numel() for p in bacformer_model.parameters() if p.requires_grad))

    # create a trainer
    # get training args
    output_dir = "output/gene_essentiality_pred"
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
        metric_for_best_model="eval_macro_auroc",
        load_best_model_at_end=True,
        greater_is_better=True,
    )

    # define a collate function for the dataset
    collate_genome_samples_fn = partial(collate_genome_samples, SPECIAL_TOKENS_DICT["PAD"], 8000, 1000)
    trainer = BacformerTrainer(
        model=bacformer_model,
        data_collator=collate_genome_samples_fn,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        compute_metrics=compute_metrics_gene_essentiality_pred,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # train the model
    trainer.train()

    # evaluate the model on the validation set
    val_output = trainer.predict(dataset["validation"])
    print("Validation output:", val_output.metrics)
    test_output = trainer.predict(dataset["test"])
    print("Test output:", test_output.metrics)

    # investigate the predictions
    test_output.predictions.squeeze(-1)
    print(test_output.predictions.shape)
    print(test_output.label_ids.shape)

    torch.save(
        {"predictions": test_output.predictions.squeeze(-1), "labels": test_output.label_ids},
        "test_preds_and_labels.pt",
    )
    # save the predictions and labels

    # plot the distribution of the predictions for a given genome of essential vs non-essential genes

    # use the trained model to predict the trait on a single new genome using an assembly


if __name__ == "__main__":
    run()
