import logging

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import LabelEncoder


def leiden_clustering(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    n_neighbors: int = 10,
    resolution: float = 1.0,
) -> anndata.AnnData:
    """
    Create an AnnData object from embeddings (X) and metadata, then run Leiden clustering.

    Parameters
    ----------
    embeddings : np.ndarray of shape (n_samples, n_features)
        Embedding matrix for your samples.
    metadata : dict or pd.DataFrame
        Per-genome metadata.
        Length of metadata must match n_samples in `embeddings`.
    resolution : float, default=1.0
        Resolution parameter for Leiden clustering.
    n_neighbors : int, default=10
        Number of neighbors to construct the neighborhood graph in Scanpy.

    Returns
    -------
    adata : anndata.AnnData
        The AnnData object containing embeddings and clustering results.
    """
    # 1. Create AnnData object from embeddings
    adata = anndata.AnnData(X=embeddings.copy())

    # Make sure 'metadata' is a pandas DataFrame if it's a dict
    if isinstance(metadata, dict):
        metadata = pd.DataFrame(metadata)
    elif not isinstance(metadata, pd.DataFrame):
        raise ValueError("metadata must be either a dict or pandas DataFrame.")

    # Assign metadata to adata.obs
    # (Ensure the length of metadata matches the number of rows in `embeddings`)
    adata.obs = metadata.reset_index(drop=True)

    # 2. Compute nearest neighbors and run Leiden clustering
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")  # use the 'X' matrix as input
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden_clusters")
    return adata


def compute_leiden_clustering_metrics(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    n_neighbors: int = 10,
    resolution: float = 1.0,
    label_key: str = "species",
) -> tuple[anndata.AnnData, float, float, float]:
    """
    Create an AnnData object from embeddings (X) and metadata, then run Leiden clustering.

    Parameters
    ----------
    embeddings : np.ndarray of shape (n_samples, n_features)
        Embedding matrix for your samples.
    metadata : dict or pd.DataFrame
        Per-genome metadata.
        Length of metadata must match n_samples in `embeddings`.
    resolution : float, default=1.0
        Resolution parameter for Leiden clustering.
    n_neighbors : int, default=10
        Number of neighbors to construct the neighborhood graph in Scanpy.

    Returns
    -------
    adata : anndata.AnnData
        The AnnData object containing embeddings and clustering results.
    ari : float
        Adjusted Rand Index.
    nmi : float
        Normalized Mutual Information.
    sil : float
        Silhouette Score.
    """
    adata = leiden_clustering(
        embeddings=embeddings,
        metadata=metadata,
        n_neighbors=n_neighbors,
        resolution=resolution,
    )

    # Convert Leiden cluster labels to integer labels
    leiden_clusters = adata.obs["leiden_clusters"].astype(int)

    # 3. Encode your ground-truth labels
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(adata.obs[label_key])

    # 4. Compute ARI, NMI, and Silhouette
    ari = adjusted_rand_score(numeric_labels, leiden_clusters)
    nmi = normalized_mutual_info_score(numeric_labels, leiden_clusters)
    # Silhouette requires sample-level features + predicted labels
    sil = silhouette_score(adata.X, leiden_clusters)

    logging.info(f"Leiden clustering at resolution={resolution}")
    logging.info(f"  Adjusted Rand Index (ARI): {ari}")
    logging.info(f"  Normalized Mutual Information (NMI): {nmi}")
    logging.info(f"  Silhouette Score: {sil}")

    return adata, ari, nmi, sil
