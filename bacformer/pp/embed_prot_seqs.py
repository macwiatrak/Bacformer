import logging
import re
from collections.abc import Callable
from typing import Any, Literal

from bacbench.modeling.utils import protein_embeddings_to_inputs
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer

try:
    from faesm.esm import FAEsmForMaskedLM
    from faesm.esmc import ESMC

    faesm_installed = True
except ImportError:
    faesm_installed = False
    logging.warning(
        "faESM (fast ESM) not installed, this will lead to significant slowdown. "
        "Defaulting to use HuggingFace implementation. "
        "Please consider installing faESM: https://github.com/pengzhangzhi/faplm"
    )


import numpy as np
import pandas as pd
import torch


def average_unpadded(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average unpadded token embeddings across sequences. Built for ESMC.

    Args:
      embeddings: (N, D) unpadded token embeddings concatenated across B sequences
      attention_mask: (B, T) indicating which tokens in each sequence are real (1) vs pad (0).
                     The total number of real tokens across all B sequences should be N.

    Returns
    -------
      (B, D) tensor of per-sequence average embeddings (excluding pad).
    """
    B, T = attention_mask.shape  # e.g. (2, 24)

    # 1) Compute unpadded lengths for each sequence
    #    e.g. if attention_mask[1] has 21 tokens == 1, it means 21 unpadded tokens for seq #1
    lengths = attention_mask.sum(dim=1)  # (B,)

    # 2) Slice the embeddings for each sequence
    #    We assume the embeddings have been "unpadded" and concatenated in order:
    #    first all tokens from seq 0, then seq 1, etc.
    results = []
    start_idx = 0
    for i in range(B):
        # number of tokens in sequence i
        seq_len = lengths[i].item()

        # slice out embeddings for sequence i
        seq_emb = embeddings[start_idx : start_idx + seq_len]  # shape (seq_len, D)
        start_idx += seq_len

        # average across tokens
        seq_avg = seq_emb.mean(dim=0)  # (D,)
        results.append(seq_avg)

    # 3) Stack results -> (B, D)
    return torch.stack(results, dim=0)


def generate_protein_embeddings(
    model: Callable,
    tokenizer: Callable,
    protein_sequences: list[str],
    model_type: Literal["esm2", "esmc", "protbert"],
    batch_size: int = 64,
    max_seq_len: int = 1024,
) -> list[np.ndarray]:
    """Generate protein embeddings using pretrained models.

    Args:
        model (Callable): The pretrained model to use for generating embeddings.
        protein_sequences (List[str]): List of protein sequences to generate embeddings for.
        model_type (str): Type of the model, either "esm2" or "esmc".
        batch_size (int): Batch size for processing sequences.
        max_seq_len (int): Maximum sequence length for the model.
    :return: List[np.ndarray]: List of protein embeddings.
    """
    # Initialize an empty list to store the protein embeddings
    mean_protein_embeddings = []

    # get model device
    device = model.device

    # Process the protein sequences in batches
    for i in range(0, len(protein_sequences), batch_size):
        batch_sequences = protein_sequences[i : i + batch_size]

        if model_type == "protbert":
            # process the sequences into protbert format
            batch_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in batch_sequences]

        # Generate embeddings for the current batch
        inputs = tokenizer(
            batch_sequences,
            add_special_tokens=True,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_seq_len,
        )
        # move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get the last hidden state from the model
        with torch.no_grad():
            if model_type == "esm2":
                last_hidden_state = model(**inputs)["last_hidden_state"]
                # Get protein representations from amino acid token representations
                protein_representations = torch.einsum(
                    "ijk,ij->ik", last_hidden_state, inputs["attention_mask"].type_as(last_hidden_state)
                ) / inputs["attention_mask"].sum(1).unsqueeze(1)
            elif model_type == "esmc":
                last_hidden_state = model(inputs["input_ids"]).embeddings
                # Get protein representations from amino acid token representations
                protein_representations = average_unpadded(last_hidden_state, inputs["attention_mask"])
            elif model_type == "protbert":
                last_hidden_state = model(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                ).last_hidden_state
                protein_representations = torch.einsum(
                    "ijk,ij->ik", last_hidden_state, inputs["attention_mask"].type_as(last_hidden_state)
                ) / inputs["attention_mask"].sum(1).unsqueeze(1)

        # Append the generated embeddings to the list, moving them to CPU and converting to numpy
        mean_protein_embeddings += list(protein_representations.cpu().numpy())

    return mean_protein_embeddings


def compute_genome_protein_embeddings(
    model: Callable,
    tokenizer: Callable,
    protein_sequences: list[str] | list[list[str]],
    contig_ids: list[str] = None,
    model_type: Literal["esm2", "esmc", "protbert"] = "esm2",
    batch_size: int = 64,
    max_prot_seq_len: int = 1024,
    genome_pooling_method: Literal["mean", "max"] = None,
) -> list[np.ndarray] | np.ndarray:
    """Embed genome protein sequences using pretrained models.

    Args:
        protein_sequences (List[str]): List or list of lists of protein sequences to generate embeddings for.
        model_type (str): Type of the model, either "esm2" or "esmc".
        batch_size (int): Batch size for processing sequences.
        max_seq_len (int): Maximum sequence length for the model.
        protein_pooling_method (str): Pooling method to use on protein level, either "mean" or "cls".
        genome_pooling_method (str): Pooling method to use on genome level, either "mean" or "cls".
    :return: List[np.ndarray]: List of protein embeddings.
    """
    assert len(protein_sequences) > 0, "Protein sequence list is empty, please include proteins in the list"

    # if the list of protein sequences is not nested, make it nested
    if isinstance(protein_sequences[0], str):
        protein_sequences = [protein_sequences]

    if contig_ids is not None:
        assert len(protein_sequences) == len(contig_ids), "Length of protein sequences and contig IDs must match"
    else:
        # create dummy contig ids to make it work in the next step
        contig_ids = [0] * len(protein_sequences)

    # create and explode dataframe
    prot_seqs_df = pd.DataFrame(
        {
            "contig_id": contig_ids,
            "protein_sequence": protein_sequences,
        }
    )
    # get contig order which will be useful later
    prot_seqs_df["contig_idx"] = range(len(prot_seqs_df))
    prot_seqs_df = prot_seqs_df.explode("protein_sequence")
    # get protein order which will be useful later
    prot_seqs_df["protein_index"] = range(len(prot_seqs_df))
    # get protein sequence length
    prot_seqs_df["prot_len"] = prot_seqs_df["protein_sequence"].apply(len)
    # sort by protein length, this is important for the model inference speedup
    prot_seqs_df = prot_seqs_df.sort_values(by="prot_len")

    # embed protein sequences
    protein_embeddings = generate_protein_embeddings(
        model=model,
        tokenizer=tokenizer,
        protein_sequences=prot_seqs_df["protein_sequence"].tolist(),
        model_type=model_type,
        batch_size=batch_size,
        max_seq_len=max_prot_seq_len,
    )

    # if we pool all the embeddings at genome level, we don't care about the order and we just
    # pool and return avg embeddings
    if genome_pooling_method is not None:
        protein_embeddings = np.stack(protein_embeddings)
        if genome_pooling_method == "mean":
            return np.mean(protein_embeddings, axis=0)
        if genome_pooling_method == "max":
            return np.max(protein_embeddings, axis=0)
        raise ValueError(f"Unsupported genome pooling method: {genome_pooling_method}")

    # if we pool at protein level, we need to return the embeddings in the same order as the input
    prot_seqs_df["protein_embedding"] = protein_embeddings
    # sort by protein index
    prot_seqs_df = prot_seqs_df.sort_values(by="protein_index")
    # group by contig id and get the list of protein embeddings
    prot_seqs_df = prot_seqs_df.groupby(["contig_id", "contig_idx"])["protein_embedding"].apply(list).reset_index()
    # sort by contig index and drop it, as it is not needed anymore
    prot_seqs_df = prot_seqs_df.sort_values(by="contig_idx").drop(columns=["contig_idx"])
    # convert to list of lists
    protein_embeddings = prot_seqs_df["protein_embedding"].tolist()
    return protein_embeddings


def load_plm(
    model_path: str = "facebook/esm2_t12_35M_UR50D", model_type: Literal["esm2", "esmc", "protbert"] = "esm2"
) -> tuple[Callable, Callable]:
    """Load specified pLM."""
    if model_type.lower() not in ["esm2", "esmc", "protbert"]:
        raise ValueError("Model currently not supported, please choose out of available models: ['esm2', 'esmc']")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type.lower() == "esm2":
        if faesm_installed:
            model = FAEsmForMaskedLM.from_pretrained(model_path).to(device).eval().to(torch.float16)
            tokenizer = model.tokenizer
        else:
            model = AutoModel.from_pretrained(model_path).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif model_type.lower() == "esmc":
        if not faesm_installed:
            raise ValueError(
                "ESMC only supported with faESM. Please consider installing faESM: https://github.com/pengzhangzhi/faplm"
            )
        model = ESMC.from_pretrained(model_path, use_flash_attn=True).to(device).eval().to(torch.float16)
        tokenizer = model.tokenizer
    else:
        # load protbert
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)

    return model, tokenizer


def compute_bacformer_embeddings(
    model: Callable,
    protein_embeddings: list[list[np.ndarray]] | list[np.ndarray],
    contig_ids: list[str] = None,
    max_n_proteins: int = 9000,
    max_n_contigs: int = 1000,
    genome_pooling_method: Literal["mean", "max"] = None,
    prot_emb_idx: int = 4,
) -> np.ndarray:
    """Compute Bacformer embeddings for a list of protein embeddings.

    Args:
        model (BacformerModel): The Bacformer model to use for embedding.
        protein_embeddings (List[List[np.ndarray]]): The protein embeddings to compute the Bacformer embeddings for.
        max_n_proteins (int): The maximum number of proteins to use for each genome.
        max_n_contigs (int): The maximum number of contigs to use for each genome.
        genome_pooling_method (str): The pooling method to use for the genome level embedding.

    Returns
    -------
        List[np.ndarray]: The Bacformer embeddings for the input protein embeddings.
    """
    assert len(protein_embeddings) > 0, "Protein sequence list is empty, please include proteins in the list"

    # if the list of protein sequences is not nested, make it nested
    if isinstance(protein_embeddings[0], np.ndarray):
        protein_embeddings = [protein_embeddings]

    if contig_ids is not None:
        assert len(protein_embeddings) == len(contig_ids), "Length of protein sequences and contig IDs must match"
    else:
        # create dummy contig ids to make it work in the next step
        contig_ids = [0] * len(protein_embeddings)

    # create and explode dataframe
    prot_embs_df = pd.DataFrame(
        {
            "contig_id": contig_ids,
            "protein_embedding": protein_embeddings,
        }
    )
    # get contig order which will be useful later
    prot_embs_df["contig_idx"] = range(len(prot_embs_df))
    prot_embs_df = prot_embs_df.explode("protein_embedding")

    # get model inputs
    device = model.device
    inputs = protein_embeddings_to_inputs(
        protein_embeddings=protein_embeddings,
        max_n_proteins=max_n_proteins,
        max_n_contigs=max_n_contigs,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Compute Bacformer embeddings
    with torch.no_grad():
        bacformer_embeddings = model(
            protein_embeddings=inputs["protein_embeddings"].type(model.dtype),
            special_tokens_mask=inputs["special_tokens_mask"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        ).last_hidden_state

    # perform genome pooling
    if genome_pooling_method == "mean":
        return bacformer_embeddings.mean(dim=1).type(torch.float32).cpu().squeeze().numpy()
    elif genome_pooling_method == "max":
        return bacformer_embeddings.max(dim=1).values.type(torch.float32).cpu().squeeze().numpy()

    # only keep the protein embeddings and not special tokens
    bacformer_embeddings = bacformer_embeddings[inputs["special_tokens_mask"] == prot_emb_idx]
    # make it into a list
    bacformer_embeddings = list(bacformer_embeddings.type(torch.float32).cpu().numpy())

    # sort the embeddings by the protein index
    prot_embs_df["protein_embedding"] = bacformer_embeddings
    # group by contig id and get the list of protein embeddings
    prot_embs_df = prot_embs_df.groupby(["contig_id", "contig_idx"])["protein_embedding"].apply(list).reset_index()
    # sort by contig index and drop it, as it is not needed anymore
    prot_embs_df = prot_embs_df.sort_values(by="contig_idx").drop(columns=["contig_idx"])
    # convert to list of lists
    protein_embeddings = prot_embs_df["protein_embedding"].tolist()
    return protein_embeddings


def add_protein_embeddings(
    row: dict[str, Any],
    prot_seq_col: str,
    output_col: str,
    model: Callable,
    tokenizer: Callable,
    model_type: Literal["esm2", "esmc", "protbert"] = "esm2",
    batch_size: int = 64,
    max_prot_seq_len: int = 1024,
    genome_pooling_method: Literal["mean", "max"] = None,
):
    """Helper function to add protein embeddings to a row."""
    return {
        output_col: compute_genome_protein_embeddings(
            model=model,
            tokenizer=tokenizer,
            protein_sequences=row[prot_seq_col],
            contig_ids=row.get("contig_name", None),
            model_type=model_type,
            batch_size=batch_size,
            max_prot_seq_len=max_prot_seq_len,
            genome_pooling_method=genome_pooling_method,
        )
    }


def add_bacformer_embeddings(
    row: dict[str, Any],
    input_col: str,
    output_col: str,
    model: Callable,
    max_n_proteins: int = 9000,
    max_n_contigs: int = 1000,
    genome_pooling_method: Literal["mean", "max"] = None,
) -> dict[str, Any]:
    """Helper function to add Bacformer embeddings to a row."""
    return {
        output_col: compute_bacformer_embeddings(
            model=model,
            protein_embeddings=row[input_col],
            contig_ids=row.get("contig_name", None),
            max_n_proteins=max_n_proteins,
            max_n_contigs=max_n_contigs,
            genome_pooling_method=genome_pooling_method,
        )
    }


def get_prot_seq_col_name(cols: list[str]) -> str:
    """Get the protein sequence column name from the dataframe columns.

    Args:
        cols (List[str]): The list of column names.

    Returns
    -------
        str: The protein sequence column name.
    """
    if "protein_sequence" in cols:
        return "protein_sequence"
    if "protein_sequences" in cols:
        return "protein_sequences"
    if "sequence" in cols:
        return "sequence"
    raise ValueError("No protein sequence column found in the dataframe.")


def embed_dataset_col_with_bacformer(
    dataset: Dataset | None,
    model_path: str,
    model_type: Literal["esm2", "esmc", "protbert", "bacformer"],
    batch_size: int = 64,
    max_prot_seq_len: int = 1024,
    device: str = None,
    output_col: str = "embeddings",
    genome_pooling_method: Literal["mean", "max"] = None,
    max_n_proteins: int = 9000,  # for Bacformer
    max_n_contigs: int = 1000,  # for Bacformer
):
    """Run script to embed protein sequences with various models.

    :param dataset: BacBench dataset
    :param model_path: sHuggingFace model name or path to model
    :param model_type: the used embedding model one of ["esm2", "esmc", "protbert", "bacformer"]
    :param batch_size: batch size for embedding pLMs
    :param max_prot_seq_len: max protein sequence length for embedding pLMs
    :param device: device to use for embedding pLMs, if None, will use cuda if available
    :param output_col: name of the output column for the embeddings
    :param genome_pooling_method: pooling method for the genome level embedding, one of ["mean", "max"]
    :param max_n_proteins: maximum number of proteins to use for each genome, only used for Bacformer
    :param max_n_contigs: maximum number of contigs to use for each genome, only used for Bacformer
    :param start_idx: start index for slicing the dataset, if None, will use the whole dataset
    :param end_idx: end index for slicing the dataset, if None, will use the whole dataset
    :param streaming: if True, will load the dataset in streaming mode, useful for large datasets
    :param save_every_n_rows: if set, will save the dataframe every n rows, only works for iterable datasets
    :param output_dir: output directory for saving the dataframe, only used for iterable datasets and if save_every_n_rows is set
    :return: a pandas dataframe with the protein embeddings
    """
    # set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # check if the model is Bacformer and adjust accordingly
    bacformer_model = None
    if model_type == "bacformer":
        logging.info("Bacformer model used, loading Bacformer model and its ESM-2 base model.")
        bacformer_model = (
            AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(torch.bfloat16).to(device)
        )
        model_type = "esm2"
        model_path = "facebook/esm2_t12_35M_UR50D"

    # load pLM
    model, tokenizer = load_plm(model_path, model_type)

    # embed protein sequences in the dataset
    # get the protein sequence column name
    prot_col = get_prot_seq_col_name(dataset.column_names)

    # 1) embed every protein sequence in this split
    dataset = dataset.map(
        lambda row: add_protein_embeddings(
            row=row,
            prot_seq_col=prot_col,
            output_col=output_col,
            model=model,
            tokenizer=tokenizer,
            model_type=model_type,
            batch_size=batch_size,
            max_prot_seq_len=max_prot_seq_len,
            genome_pooling_method=genome_pooling_method if bacformer_model is None else None,
        ),
        batched=False,
        remove_columns=[prot_col],
    )

    # 2) pass through Bacformer
    if bacformer_model is not None:
        dataset = dataset.map(
            lambda row: add_bacformer_embeddings(
                row=row,
                input_col=output_col,
                output_col=output_col,
                model=bacformer_model,
                max_n_proteins=max_n_proteins,
                max_n_contigs=max_n_contigs,
                genome_pooling_method=genome_pooling_method,
            ),
            batched=False,
        )

    return dataset
