import gzip
import logging
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

from bacformer.modeling.config import SPECIAL_TOKENS_DICT
from bacformer.modeling.modeling_base import BacformerModel

try:
    from faesm.esm import FAEsmModel
    from faesm.esmc import ESMC

    faesm_installed = True
except ImportError:
    faesm_installed = False
    logging.warning(
        "faESM (fast ESM) not installed, this will lead to significant slowdown. "
        "Defaulting to use HuggingFace implementation. "
        "Please consider installing faESM: https://github.com/pengzhangzhi/faplm"
    )

from Bio import SeqIO


def extract_protein_info_from_genbank(filepath: str):
    """
    Extract gene details from a GenBank file (.gbff or .gbff.gz) and return a DataFrame.

    Args:
        filepath (str): Path to the GenBank file (.gbff or .gbff.gz).

    Returns
    -------
        pd.DataFrame: A DataFrame with columns [accession, definition, gene, locus_tag, start, end, protein_id, translation].
    """
    # Handle gzip-compressed files
    if filepath.endswith(".gz"):
        handle = gzip.open(filepath, "rt")
    else:
        handle = open(filepath)

    # Parse the GenBank file
    records = SeqIO.parse(handle, "genbank")
    data = []

    contig_idx = 0
    for record in records:
        # Extract record-level metadata
        accession = record.annotations.get("accessions", [None])[0]
        definition = record.description
        genome_name = record.annotations.get("organism", None)

        for feature in record.features:
            if feature.type == "CDS":  # Only focus on coding sequences (CDS)
                # Extract gene name
                gene_name = feature.qualifiers.get("gene", [None])[0]

                # Extract locus tag
                locus_tag = feature.qualifiers.get("locus_tag", [None])[0]

                # Extract start and end
                start = int(feature.location.start)
                end = int(feature.location.end)

                # Extract protein_id
                protein_id = feature.qualifiers.get("protein_id", [None])[0]

                # Extract amino acid translation
                translation = feature.qualifiers.get("translation", [None])[0]

                # Append to data list
                if translation is None:
                    continue
                data.append(
                    {
                        "strain_name": genome_name,
                        "accession_id": accession,
                        "accession_name": definition,
                        "gene_name": gene_name,
                        "protein_name": locus_tag,
                        "start": start,
                        "end": end,
                        "protein_id": protein_id,
                        "contig_idx": contig_idx,
                        "protein_sequence": translation,
                    }
                )

        contig_idx += 1

    # Close the file handle
    handle.close()

    # Create DataFrame
    df = pd.DataFrame(data)
    return df


def extract_protein_info_from_gff(filepath):
    """
    Extract protein details from a GFF file (.gff or .gff.gz) into a pandas DataFrame.

    Args:
        filepath (str): Path to the GFF file.

    Returns
    -------
        pd.DataFrame: A DataFrame where each row is a gene, with columns for gene details.
    """
    genes = []

    # Open the file based on its extension
    if filepath.endswith(".gz"):
        open_func = gzip.open
        mode = "rt"  # Read as text
    else:
        open_func = open
        mode = "r"

    with open_func(filepath, mode) as file:
        for line in file:
            if line.startswith("#"):
                # # Extract assembly ID from the header
                # if line.startswith("##sequence-region"):
                #     assembly_id = line.split()[1]
                continue

            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue  # Skip malformed lines

            feature_type = parts[2]
            if feature_type != "CDS":
                continue  # Focus only on genes

            seqid = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attributes = parts[8]

            # Parse attributes
            attr_dict = {}
            for attr in attributes.split(";"):
                if "=" in attr:
                    key, value = attr.split("=", 1)
                    attr_dict[key] = value

            # Extract specific fields
            gene_name = attr_dict.get("gene", None)
            locus_tag = attr_dict.get("locus_tag", None)
            protein_id = attr_dict.get("protein_id", None)

            # Append gene info to the list
            genes.append(
                {
                    "seqid": seqid,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "gene_name": gene_name if gene_name is not None else locus_tag,
                    "locus_tag": locus_tag,
                    "protein_id": protein_id,
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(genes)
    return df


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


def load_esm_model(
    model_path: str = "facebook/esm2_t12_35M_UR50D", model_type: Literal["esm2", "esmc"] = "esm2"
) -> Callable:
    """Load specified ESM model."""
    if model_type.lower() not in ["esm2", "esmc"]:
        raise ValueError("Model currently not supported, please choose out of available models: ['esm2', 'esmc']")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type.lower() == "esm2":
        if faesm_installed:
            model = FAEsmModel.from_pretrained(model_path).to(device).eval().to(torch.float16)
        else:
            model = AutoModel.from_pretrained(model_path).to(device).eval()
    elif model_type.lower() == "esmc" and faesm_installed:
        model = ESMC.from_pretrained(model_path, use_flash_attn=True).to(device).eval().to(torch.float16)
    else:
        raise ValueError(
            "ESMC only supported with faESM. Please consider installing faESM: https://github.com/pengzhangzhi/faplm"
        )

    return model


def generate_protein_embeddings(
    model: Callable,
    protein_sequences: list[str],
    model_type: Literal["esm2", "esmc"],
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

        # Generate embeddings for the current batch
        inputs = model.tokenizer(
            batch_sequences, return_tensors="pt", padding="longest", truncation=True, max_length=max_seq_len
        )
        # move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get the last hidden state from the model
        with torch.no_grad():
            if model_type == "esm2":
                last_hidden_state = model(**inputs)["last_hidden_state"]
                # Get protein representations from amino acid token representations
                protein_representations = torch.einsum(
                    "ijk,ij->ik", last_hidden_state, inputs["attention_mask"].float()
                ) / inputs["attention_mask"].float().sum(1).unsqueeze(1)
            elif model_type == "esmc":
                last_hidden_state = model(inputs["input_ids"]).embeddings
                # Get protein representations from amino acid token representations
                protein_representations = average_unpadded(last_hidden_state, inputs["attention_mask"])

        # Append the generated embeddings to the list, moving them to CPU and converting to numpy
        mean_protein_embeddings += list(protein_representations.cpu().numpy())

    return mean_protein_embeddings


def embed_genome_protein_sequences(
    model: Callable,
    protein_sequences: list[str] | list[list[str]],
    contig_ids: list[str] = None,
    model_type: Literal["esm2", "esmc"] = "esm2",
    batch_size: int = 64,
    max_seq_len: int = 1024,
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
    ).explode("protein_sequence")
    # get protein order which will be useful later
    prot_seqs_df["protein_index"] = range(len(prot_seqs_df))
    # get protein sequence length
    prot_seqs_df["prot_len"] = prot_seqs_df["protein_sequence"].apply(len)
    # sort by protein length, this is important for the model inference speedup
    prot_seqs_df = prot_seqs_df.sort_values(by="prot_len")

    # embed protein sequences
    protein_embeddings = generate_protein_embeddings(
        model=model,
        protein_sequences=prot_seqs_df["protein_sequence"].tolist(),
        model_type=model_type,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
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
    prot_seqs_df = prot_seqs_df.groupby("contig_id")["protein_embedding"].apply(list).reset_index()
    # convert to list of lists
    protein_embeddings = prot_seqs_df["protein_embedding"].tolist()
    return protein_embeddings


def protein_embeddings_to_inputs(
    protein_embeddings: list[list[np.ndarray]] | list[np.ndarray],
    max_n_proteins: int = 6000,
    max_n_contigs: int = 1000,
    cls_token_id: int = SPECIAL_TOKENS_DICT["CLS"],
    sep_token_id: int = SPECIAL_TOKENS_DICT["CLS"],
    prot_emb_token_id: int = SPECIAL_TOKENS_DICT["PROT_EMB"],
    end_token_id: int = SPECIAL_TOKENS_DICT["END"],
    torch_dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Convert protein embeddings to inputs for Bacformer model.

    Args:
        protein_embeddings (List[List[np.ndarray]]): The protein embeddings to convert.
        max_n_proteins (int): The maximum number of proteins to use for each genome.
        max_n_contigs (int): The maximum number of contigs to use for each genome.
        cls_token_id (int): The ID of the CLS token.
        sep_token_id (int): The ID of the SEP token.
        prot_emb_token_id (int): The ID of the protein embedding token.
        end_token_id (int): The ID of the END token.

    Returns
    -------
        dict: The inputs for the Bacformer model.
    """
    assert len(protein_embeddings) > 0, "protein_embeddings should not be empty"

    # check if protein_embeddings is a list of lists, if not, make it one
    if not isinstance(protein_embeddings[0], list):
        protein_embeddings = [protein_embeddings]

    # preprocess protein embeddings
    dim = len(protein_embeddings[0][0])
    pad_emb = torch.zeros(dim, dtype=torch_dtype)

    special_tokens_mask = [cls_token_id]
    protein_embeddings_output = [pad_emb]
    token_type_ids = [0]

    # iterate through contigs
    for contig_idx, contig in enumerate(protein_embeddings):
        # check if contig does not exceed max_n_contigs
        if contig_idx > max_n_contigs:
            contig_idx = max_n_contigs - 1
        # iterate through prots in contig
        for prot_emb in contig:
            # append prot_emb_token_id to special tokens mask
            special_tokens_mask.append(prot_emb_token_id)
            # append prot_emb to protein_embeddings_output
            protein_embeddings_output.append(torch.tensor(prot_emb, dtype=torch_dtype))
            # append contig_idx to token_type_ids
            token_type_ids.append(contig_idx)
        # separate the contigs with a SEP token
        special_tokens_mask.append(sep_token_id)
        protein_embeddings_output.append(pad_emb)
        token_type_ids.append(contig_idx)

    # account for the END token
    protein_embeddings_output = protein_embeddings_output[: max_n_proteins - 1] + [pad_emb]
    protein_embeddings_output = torch.stack(protein_embeddings_output)

    special_tokens_mask = special_tokens_mask[: max_n_proteins - 1] + [end_token_id]
    special_tokens_mask = torch.tensor(special_tokens_mask)

    token_type_ids = token_type_ids[: max_n_proteins - 1] + [contig_idx]
    token_type_ids = torch.tensor(token_type_ids)

    attention_mask = torch.ones_like(special_tokens_mask)
    return {
        "protein_embeddings": protein_embeddings_output,
        "special_tokens_mask": special_tokens_mask,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }


def compute_bacformer_embeddings(
    model: BacformerModel,
    protein_embeddings: list[list[np.ndarray]] | list[np.ndarray],
    max_n_proteins: int = 6000,
    max_n_contigs: int = 1000,
    genome_pooling_method: Literal["mean", "max"] = None,
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
            protein_embeddings=inputs["protein_embeddings"],
            special_tokens_mask=inputs["special_tokens_mask"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        ).last_hidden_state

    # perform genome pooling
    if genome_pooling_method == "mean":
        return bacformer_embeddings.mean(dim=1).cpu().squeeze().numpy()
    elif genome_pooling_method == "max":
        return bacformer_embeddings.max(dim=1).values.cpu().squeeze().numpy()

    return bacformer_embeddings.squeeze().cpu().numpy()


def embed_genome(
    protein_sequences: list[str] | list[list[str]],
    bacformer_model: BacformerModel,
    esm_model: AutoModel | FAEsmModel,
    max_n_proteins: int = 6000,
    max_n_contigs: int = 1000,
    contig_ids: list[str] = None,
    prot_embed_batch_size: int = 64,
    max_prot_seq_len: int = 1024,
    genome_pooling_method: Literal["mean", "max"] = None,
):
    """Add docstrings"""
    # embed protein sequences with ESM model
    protein_embeddings = embed_genome_protein_sequences(
        model=esm_model,
        protein_sequences=protein_sequences,
        contig_ids=contig_ids,
        model_type="esm2",
        batch_size=prot_embed_batch_size,
        max_seq_len=max_prot_seq_len,
        genome_pooling_method=None,
    )

    # use Bacformer to compute contextualised protein representations
    bacformer_embed = compute_bacformer_embeddings(
        model=bacformer_model,
        protein_embeddings=protein_embeddings,
        max_n_proteins=max_n_proteins,
        max_n_contigs=max_n_contigs,
        genome_pooling_method=genome_pooling_method,
    )
    return bacformer_embed
