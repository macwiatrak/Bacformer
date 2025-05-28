import os
import random
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
from dataclasses_json import dataclass_json
from datasets import IterableDataset, load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed

from bacformer.modeling.config import SPECIAL_TOKENS_DICT

# masking strategy inspired by BERT where we mask 80%, leave 10% unchanged and replace 10% with random tokens
MASKING_STRATEGY = {
    "mask": 0.875,
    "no_change": 0.125,
}


@dataclass_json
@dataclass
class DataReaderOutput:
    """A dataclass for storing the output of the data reader."""

    train_dataset: IterableDataset = None
    val_dataset: IterableDataset = None
    test_dataset: IterableDataset = None


def mask_prot_embs(
    special_tokens_mask: np.ndarray,
    labels: np.ndarray,
    mgm_probability: float,
    prot_emb_token_id: int,
    mask_token_id: int,
    masking_strategy: dict[str, float] = MASKING_STRATEGY,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask protein embeddings with the given probability."""
    if mgm_probability == 0.0:
        return special_tokens_mask, labels

    mask_labels = np.ones_like(labels) * -100
    indices = np.where(special_tokens_mask == prot_emb_token_id)[0]

    # replace tokens with mask token
    n_prots_to_mask = int(mgm_probability * len(indices) * masking_strategy["mask"])
    # randomly sample proteins to mask
    indices_to_mask = np.random.choice(indices, n_prots_to_mask, replace=False)
    # replace the selected protein embeddings with the mask token
    special_tokens_mask[indices_to_mask] = mask_token_id
    mask_labels[indices_to_mask] = labels[indices_to_mask]

    # leave some tokens unchanged
    n_prots_no_change = int(mgm_probability * len(indices) * masking_strategy["no_change"])
    indices_no_change = np.random.choice(indices, n_prots_no_change, replace=False)
    mask_labels[indices_no_change] = labels[indices_no_change]

    return special_tokens_mask, mask_labels


def transform_sample(
    mgm_probability: float = 0.0,
    prot_emb_token_id: int = SPECIAL_TOKENS_DICT["PROT_EMB"],
    mask_token_id: int = SPECIAL_TOKENS_DICT["MASK"],
    sample: dict[str, Any] = None,
) -> dict[str, Any]:
    """Transform sample for the model input."""
    special_tokens_mask, labels = mask_prot_embs(
        special_tokens_mask=np.array(sample["processed_data"]["special_tokens_mask"]),
        labels=np.array(sample["processed_data"]["labels"]),
        mgm_probability=mgm_probability,
        prot_emb_token_id=prot_emb_token_id,
        mask_token_id=mask_token_id,
    )
    # overwrite the special tokens mask
    sample["processed_data"]["special_tokens_mask"] = special_tokens_mask
    sample["processed_data"]["labels"] = labels
    return sample


def fetch_training_data(
    input_dir: str,
    mgm_probability: float,
    test: bool = False,
    random_state: int = 42,
):
    """A function which orchestrates getting the data.

    The pretraining data is stored in chunks of parquet files. The function reads the parquet files and returns
    IterableDataset objects for training, validation, and test data.
    """
    # get train data from SPIRE and MGNify
    set_seed(random_state)
    # get train files
    train_files = [
        os.path.join(input_dir, "train", f)
        for f in os.listdir(os.path.join(input_dir, "train"))
        if f.endswith("parquet")
    ]
    # shuffle the files
    random.shuffle(train_files)
    data_files = {
        "train": train_files,
        "validation": [
            os.path.join(input_dir, "val", f)
            for f in os.listdir(os.path.join(input_dir, "val"))
            if f.endswith("parquet")
        ],
        "test": [
            os.path.join(input_dir, "test", f)
            for f in os.listdir(os.path.join(input_dir, "test"))
            if f.endswith("parquet")
        ],
    }
    transform_fn = partial(
        transform_sample, mgm_probability, SPECIAL_TOKENS_DICT["PROT_EMB"], SPECIAL_TOKENS_DICT["MASK"]
    )
    train_dataset = (
        load_dataset("parquet", data_files=data_files, split="train", streaming=True)
        .select_columns("processed_data")
        .map(transform_fn, batched=False, with_indices=False)
    )
    val_dataset = (
        load_dataset("parquet", data_files=data_files, split="validation", streaming=True)
        .select_columns("processed_data")
        .map(transform_fn, batched=False, with_indices=False)
    )

    if not test:
        return DataReaderOutput(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

    test_dataset = (
        load_dataset("parquet", data_files=data_files, split="test", streaming=True)
        .select_columns("processed_data")
        .map(transform_fn, batched=False, with_indices=False)
    )
    return DataReaderOutput(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )


def collate_genome_samples(
    pad_token_id: int = 0,
    max_n_proteins: int = 9000,
    max_n_contigs: int = 1000,
    samples: list[dict] = None,
) -> dict[str, torch.Tensor]:
    """Collate function for GenomeSample."""
    prot_emb = pad_sequence(
        [
            torch.tensor(sample["protein_embeddings"], dtype=torch.float32).squeeze(0)[:max_n_proteins]
            for sample in samples
        ],
        batch_first=True,
        padding_value=pad_token_id,
    )
    special_tokens_mask = pad_sequence(
        [
            torch.tensor(sample["special_tokens_mask"], dtype=torch.long).squeeze(0)[:max_n_proteins]
            for sample in samples
        ],
        batch_first=True,
        padding_value=pad_token_id,
    )
    token_type_ids = pad_sequence(
        [torch.tensor(sample["token_type_ids"], dtype=torch.long).squeeze(0)[:max_n_proteins] for sample in samples],
        batch_first=True,
        padding_value=max_n_contigs,
    )

    if "labels" in samples[0]:
        labels = pad_sequence(
            [torch.tensor(sample["labels"], dtype=torch.long).squeeze(0)[:max_n_proteins] for sample in samples],
            batch_first=True,
            padding_value=-100,
        )
    elif "label" in samples[0]:
        labels = torch.tensor([sample["label"] for sample in samples], dtype=torch.long)
    else:
        labels = torch.tensor([])

    output = {
        "protein_embeddings": prot_emb,
        "special_tokens_mask": special_tokens_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
    }
    if "attention_mask" in samples[0]:
        padding_mask = pad_sequence(
            [
                torch.tensor(sample["attention_mask"], dtype=torch.float32).squeeze(0)[:max_n_proteins]
                for sample in samples
            ],
            batch_first=True,
            padding_value=pad_token_id,
        )
        output["attention_mask"] = padding_mask

    return output
