import numpy as np


def operon_prot_indices_to_pairwise_labels(operon_prot_indices: list[list[int]], n_genes: int) -> np.ndarray:
    """Convert operon protein indices to pairwise binary labels across whole sequence.

    Parameters
    ----------
    operon_prot_indices : list of operons; each operon is a list of 0-based gene indices
    n_genes  : total number of genes (N)

    Returns
    -------
    np.ndarray  (shape = (N-1,))
        binary[i] = 1  ⇔  genes i and i+1 are in the SAME operon
    """
    binary = np.zeros(n_genes - 1, dtype=int)

    for operon in operon_prot_indices:
        max_operon = max(operon)
        min_operon = min(operon)
        # mark every adjacent pair inside that operon
        for item in range(min_operon, max_operon):
            binary[item] = 1

    return binary


def get_intergenic_bp_dist(starts: list[int], ends: list[int]) -> np.ndarray:
    """Compute intergenic distances in base pairs between genes."""
    out = []
    for idx in range(len(starts) - 1):
        d = starts[idx + 1] - ends[idx]
        out.append(d)
    return np.array(out)


def predict_pairwise_operon_boundaries(
    emb: np.ndarray,
    intergenic_bp: np.ndarray,
    strand: np.ndarray,
    scale_bp: int = 500,
    max_gap: int = 500,
) -> np.ndarray:
    """Predict pairwise operon boundaries based on embeddings and intergenic distances.

    params
    :param emb: Embeddings of the (avg) protein sequences, shape (n, d).
    :param intergenic_bp: Intergenic distances in base pairs, shape (n-1,).
    :param strand: Strand information, shape (n,).
    :param scale_bp: Scaling factor for intergenic distances, default 500.
    :param max_gap: Maximum gap allowed for operon prediction, default 500.

    :return: predicted operon boundary scores
    """
    emb_n = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cos = np.sum(emb_n[:-1] * emb_n[1:], axis=1).reshape(-1, 1)
    cos = (cos + 1) / 2

    same = (strand[:-1] == strand[1:]).astype(int)
    diff_mask = same == 0

    # exponential distance weighting
    if scale_bp is not None:
        cos[:, 0] *= np.exp(-intergenic_bp / scale_bp)

    cos[diff_mask] = 0.0  # strand veto

    # ---------- NEW distance-gate veto ------------------------------------
    if max_gap is not None:
        cos[intergenic_bp > max_gap] = 0.0  # hard boundary

    return cos[:, 0]
