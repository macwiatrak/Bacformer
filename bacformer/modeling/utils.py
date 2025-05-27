import logging
import os

import numpy as np
import torch
from torch.nn.functional import cross_entropy, softmax
from torchmetrics.functional import accuracy, auroc, average_precision, f1_score, precision, recall
from transformers import EvalPrediction

from bacformer.modeling.config import SPECIAL_TOKENS_DICT


def create_4d_from_2d_attn_mask(attn_mask: torch.Tensor, num_attn_heads: int):
    """Helper function to reshape attn_mask to 3D from 2D"""
    assert len(attn_mask.shape) == 2, (
        f"Please provide attn_mask of shape (batch_size, seq_len), current shape {attn_mask.shape}"
    )

    bs, seq_len = attn_mask.shape
    attn_mask = attn_mask.view(bs, 1, 1, seq_len)
    attn_mask = attn_mask.expand(-1, num_attn_heads, -1, -1)
    attn_mask = attn_mask.view(bs, num_attn_heads, -1, seq_len)
    return attn_mask


def get_gpu_info() -> tuple[int, bool]:
    """Helper function to see if we are training on GPU, XPU or CPU"""
    try:
        # get nr of XPUs if training on Intel GPUs
        n_gpus = torch.xpu.device_count()
        logging.info(f"Nr of XPU devices available: {n_gpus}")
        if n_gpus > 0:
            return n_gpus, True
        n_gpus = torch.cuda.device_count()
        logging.info(f"Nr of CUDA devices available: {n_gpus}")
        if n_gpus > 0:
            return n_gpus, False
    except AttributeError:
        n_gpus = torch.cuda.device_count()
        logging.info(f"Nr of GPU devices available: {n_gpus}")
        return n_gpus, False


def find_ckpt_in_dir(ckpt_dir: str) -> str | None:
    """Find the checkpoint in the directory."""
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint")]
    if len(ckpts) == 0:
        return None
    max_idx = np.argmax([int(ckpt.split("-")[-1]) for ckpt in ckpts])
    ckpt = os.path.join(ckpt_dir, ckpts[max_idx])
    print("Using checkpoint:", ckpt)
    return ckpt


def pretraining_metrics_fn(preds: EvalPrediction) -> dict:
    """Compute metrics for the prediction step."""
    metrics = {key: val.mean().item() for key, val in preds.predictions.items()}
    return metrics


def compute_contrastive_loss(
    protein_embeddings: torch.Tensor,
    last_hidden_state: torch.Tensor,
    special_tokens_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute contrastive loss between protein embeddings and masked items."""
    # keep protein embeddings and masked items
    # ensure the batch size is 1, the model currently does not work with batch size > 1
    assert protein_embeddings.shape[0] == last_hidden_state.shape[0] == 1

    # subset to mask and protein embedding tokens
    special_tokens_mask = special_tokens_mask.squeeze(0)
    mask = (special_tokens_mask == SPECIAL_TOKENS_DICT["PROT_EMB"]) | (
        special_tokens_mask == SPECIAL_TOKENS_DICT["MASK"]
    )
    protein_embeddings = protein_embeddings.squeeze(0)[mask]
    last_hidden_state = last_hidden_state.squeeze(0)[mask]

    # Normalize embeddings
    last_hidden_state = last_hidden_state / last_hidden_state.norm(dim=1, keepdim=True)
    protein_embeddings = protein_embeddings / protein_embeddings.norm(dim=1, keepdim=True)

    # Compute similarity matrix and loss as before
    similarity_matrix = torch.matmul(last_hidden_state, protein_embeddings.T)

    n_prots = protein_embeddings.shape[0]
    labels = torch.arange(n_prots).to(protein_embeddings.device)

    # Compute the loss
    loss = cross_entropy(similarity_matrix, labels)
    return loss


def top_k_filtering(logits: torch.Tensor, top_k: int = 50):
    """
    Keep only top_k logits and set the rest to -inf.

    Args:
        logits (torch.Tensor): Logits of shape (batch_size, vocab_size).
        top_k (int): The number of highest probability logits to keep.

    Returns
    -------
        torch.Tensor: Filtered logits where only the top k values remain, and all others are -inf.
    """
    if top_k <= 0:
        return logits

    # Find top_k values
    top_k = min(top_k, logits.size(-1))
    vals, idx = torch.topk(logits, top_k, dim=-1)
    # Get the smallest logit in the top_k
    min_vals = vals[:, -1].unsqueeze(-1)
    # Mask all logits that are < this min value
    mask = logits < min_vals
    logits[mask] = float("-inf")
    return logits


def top_p_filtering(logits: torch.Tensor, top_p: float = 0.9):
    """
    Keep the smallest set of logits whose cumulative probability >= top_p.

    Args:
        logits (torch.Tensor): Logits of shape (batch_size, vocab_size).
        top_p (float): Cumulative probability threshold.

    Returns
    -------
        torch.Tensor: Filtered logits where only tokens within the top_p cumulative
                      probability mass are kept; the rest are set to -inf.
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

    # Identify where cumulative probability exceeds top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the mask to ensure we always keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Scatter to replicate the mask in the original ordering
    for i in range(logits.size(0)):
        remove_indices = sorted_indices[i, sorted_indices_to_remove[i]]
        logits[i, remove_indices] = float("-inf")

    return logits


def compute_metrics_gene_essentiality_pred(preds: EvalPrediction, ignore_index: int = -100, prefix: str = "eval"):
    """Compute metrics for the prediction step of gene essentiality task."""
    logits = torch.tensor(preds.predictions).squeeze(-1)
    labels = torch.tensor(preds.label_ids)
    # delete to save space
    del preds
    # compute metrics
    acc = accuracy(
        logits,
        labels,
        task="binary",
        ignore_index=ignore_index,
    )
    f1 = f1_score(
        logits,
        labels,
        task="binary",
        ignore_index=ignore_index,
    )
    prec = precision(logits, labels, task="binary", ignore_index=ignore_index)
    rec = recall(logits, labels, task="binary", ignore_index=ignore_index)
    auroc_val = auroc(logits, labels, task="binary", ignore_index=ignore_index)
    auprc = average_precision(logits, labels, task="binary", ignore_index=ignore_index)

    macro_auroc_val = torch.tensor(
        [
            auroc(logits[idx, :], labels[idx, :], task="binary", ignore_index=ignore_index)
            for idx in range(logits.shape[0])
        ]
    ).median()
    macro_auprc = torch.tensor(
        [
            average_precision(logits[idx, :], labels[idx, :], task="binary", ignore_index=ignore_index)
            for idx in range(logits.shape[0])
        ]
    ).median()

    return {
        f"{prefix}_macro_auroc": macro_auroc_val,
        f"{prefix}_macro_auprc": macro_auprc,
        f"{prefix}_auroc": auroc_val,
        f"{prefix}_auprc": auprc,
        f"{prefix}_accuracy": acc,
        f"{prefix}_f1": f1,
        f"{prefix}_precision": prec,
        f"{prefix}_recall": rec,
    }


def compute_metrics_binary_genome_pred(preds: EvalPrediction, ignore_index: int = -100, prefix: str = "eval"):
    """Compute metrics for the prediction step of predicting genome phenotype."""
    logits = torch.tensor(preds.predictions).squeeze(-1)
    labels = torch.tensor(preds.label_ids)
    # delete to save space
    del preds
    # compute metrics
    acc = accuracy(
        logits,
        labels,
        task="binary",
        ignore_index=ignore_index,
    )
    f1 = f1_score(
        logits,
        labels,
        task="binary",
        ignore_index=ignore_index,
    )
    auroc_val = auroc(logits, labels, task="binary", ignore_index=ignore_index)
    auprc = average_precision(logits, labels, task="binary", ignore_index=ignore_index)

    return {
        f"{prefix}_auroc": auroc_val,
        f"{prefix}_auprc": auprc,
        f"{prefix}_accuracy": acc,
        f"{prefix}_f1": f1,
    }
