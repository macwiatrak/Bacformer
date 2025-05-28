import os
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from bacformer.modeling import (
    SPECIAL_TOKENS_DICT,
    BacformerTrainer,
    adjust_prot_labels,
    collate_genome_samples,
    compute_metrics_gene_essentiality_pred,
)
from bacformer.pp import dataset_col_to_bacformer_inputs
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForTokenClassification, EarlyStoppingCallback, TrainingArguments


def run():
    """Train the gene essentiality prediction model."""
    # load the dataset
    dataset = load_dataset("macwiatrak/bacbench-essential-genes-protein-sequences")

    # embed the protein sequences with the ESM-2 base model and map the labels
    for split_name in dataset.keys():
        dataset[split_name] = dataset_col_to_bacformer_inputs(
            dataset=dataset[split_name],
            protein_sequences_col="sequence",
            max_n_proteins=7000,
        )
        # map the essentiality labels to a binary format
        dataset[split_name] = dataset[split_name].map(
            lambda row: adjust_prot_labels(
                labels=row["essential"],
                special_tokens=row["special_tokens_mask"],
            ),
            batched=False,
        )

    # load the Bacformer model for protein classification
    # for this task we use the Bacformer model trained on masked complete genomes
    # with a token (here protein) classification head
    config = AutoConfig.from_pretrained("macwiatrak/bacformer-masked-complete-genomes", trust_remote_code=True)
    config.num_labels = 1
    config.problem_type = "binary_classification"
    bacformer_model = AutoModelForTokenClassification.from_pretrained(
        "macwiatrak/bacformer-masked-complete-genomes", config=config, trust_remote_code=True
    ).to(torch.bfloat16)
    print("Nr of parameters:", sum(p.numel() for p in bacformer_model.parameters()))
    print("Nr of trainable parameters:", sum(p.numel() for p in bacformer_model.parameters() if p.requires_grad))

    # define the output directory for the model and metrics
    output_dir = "output/gene_essentiality_pred"
    os.makedirs(output_dir, exist_ok=True)

    # create a trainer
    # get training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=0.00015,
        num_train_epochs=100,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        seed=1,
        dataloader_num_workers=4,
        bf16=True,
        metric_for_best_model="eval_macro_auroc",
        load_best_model_at_end=True,
        greater_is_better=True,
    )

    # define a collate function for the dataset
    collate_genome_samples_fn = partial(collate_genome_samples, SPECIAL_TOKENS_DICT["PAD"], 7000, 1000)
    trainer = BacformerTrainer(
        model=bacformer_model,
        data_collator=collate_genome_samples_fn,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        compute_metrics=compute_metrics_gene_essentiality_pred,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    # train the model, takes around ~15 min on a single A100 GPU
    trainer.train()

    # evaluate the model on the test set
    test_output = trainer.predict(dataset["test"])
    print("Test output:", test_output.metrics)

    # get the predictions and labels for a single genome from the test set
    preds_strain = torch.sigmoid(torch.tensor(test_output.predictions.squeeze(-1)))[0]
    labels_strain = torch.tensor(test_output.label_ids)[0]
    genome_id = dataset["test"][0]["genome_id"]

    # make DF for plotting
    df = pd.DataFrame({"probability": preds_strain.tolist(), "label": labels_strain.tolist()})
    # remove the ignore index rows
    df = df[df.label != -100]

    # Create the KDE plot
    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        data=df[df["label"] == 0], x="probability", fill=True, color="blue", alpha=0.6, label="Non-essential genes"
    )
    sns.kdeplot(
        data=df[df["label"] == 1], x="probability", fill=True, color="goldenrod", alpha=0.6, label="Essential genes"
    )

    # Add legend and labels
    plt.title(f"Gene Essentiality Prediction for {genome_id}", fontsize=18)
    plt.legend(fontsize=20, title_fontsize=20, frameon=False, loc="upper left")
    plt.xlabel("", fontsize=12)
    plt.ylabel("", fontsize=12)
    plt.title("", fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(-0.1, 1.1)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # TODO: use the trained model to predict the trait on a single new genome using an assembly


if __name__ == "__main__":
    run()
