import numpy as np
import pandas as pd
import pytest
from bacformer.tl.clustering import compute_leiden_clustering_metrics, leiden_clustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


@pytest.fixture
def dummy_data():
    """
    Returns a dictionary containing dummy embeddings and metadata.
    """
    np.random.seed(42)
    n_samples = 20
    n_features = 5

    # Create a simple embedding matrix
    embeddings = np.random.rand(n_samples, n_features)

    # Create metadata with one categorical column "species"
    species_choices = ["cat", "dog"]
    metadata = pd.DataFrame(
        {
            "species": np.random.choice(species_choices, size=n_samples),
        }
    )

    return {"embeddings": embeddings, "metadata": metadata}


def test_leiden_clustering(dummy_data):
    """
    Test the leiden_clustering function with dummy data.
    """
    embeddings = dummy_data["embeddings"]
    metadata = dummy_data["metadata"]

    # Call the function
    adata = leiden_clustering(embeddings=embeddings, metadata=metadata, n_neighbors=5, resolution=0.5)

    # Check the returned AnnData object
    assert adata.shape == embeddings.shape, "adata shape must match embeddings shape"
    assert "leiden_clusters" in adata.obs.columns, "Leiden clusters should be in adata.obs"

    # Check that the cluster labels are strings or categorical
    # (By default, .astype(int) was used in compute_leiden_clustering_metrics, but
    #  the raw output from sc.tl.leiden is typically string categories.)
    assert adata.obs["leiden_clusters"].dtype.name in ["category", "object"], (
        "Leiden cluster labels should be stored as category or object in adata.obs"
    )


def test_leiden_clustering_mismatched_metadata():
    """
    Test that leiden_clustering raises a ValueError if metadata length mismatches embeddings.
    """
    # Create embeddings for 10 samples
    embeddings = np.random.rand(10, 3)
    # Create metadata for 8 samples
    metadata = pd.DataFrame({"species": ["cat"] * 8})

    with pytest.raises(ValueError):
        # Intentionally pass in the wrong type or mismatched length
        _ = leiden_clustering(embeddings, metadata=metadata.head(2))  # Force mismatch length


def test_compute_leiden_clustering_metrics(dummy_data):
    """
    Test compute_leiden_clustering_metrics function with dummy data.
    """
    embeddings = dummy_data["embeddings"]
    metadata = dummy_data["metadata"]

    adata, ari, nmi, sil = compute_leiden_clustering_metrics(
        embeddings=embeddings,
        metadata=metadata,
        n_neighbors=5,
        resolution=0.5,
        label_key="species",
    )

    # Basic checks
    assert adata.shape == embeddings.shape, "adata shape must match embeddings shape"
    assert "leiden_clusters" in adata.obs.columns, "Leiden clusters should be in adata.obs"

    # ARI, NMI, Silhouette are floats
    assert isinstance(ari, float), "ARI should be a float"
    assert isinstance(nmi, float), "NMI should be a float"
    assert isinstance(sil, float), "Silhouette should be a float"

    # NMI should be within [0, 1]
    assert 0.0 <= nmi <= 1.0, "NMI should be between 0 and 1"

    # ARI can be negative but typically is [-1, 1]
    assert -1.0 <= ari <= 1.0, "ARI should be between -1 and 1"

    # Silhouette is between -1 and 1
    assert -1.0 <= sil <= 1.0, "Silhouette score should be between -1 and 1"

    # (Optional) verify we can replicate metrics on the same data
    # by re-computing them manually:
    leiden_clusters = adata.obs["leiden_clusters"].astype("category").cat.codes
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    numeric_labels = le.fit_transform(adata.obs["species"])

    ari_manual = adjusted_rand_score(numeric_labels, leiden_clusters)
    nmi_manual = normalized_mutual_info_score(numeric_labels, leiden_clusters)
    sil_manual = silhouette_score(embeddings, leiden_clusters)

    assert np.isclose(ari, ari_manual), "ARI mismatch with manual calculation"
    assert np.isclose(nmi, nmi_manual), "NMI mismatch with manual calculation"
    assert np.isclose(sil, sil_manual), "Silhouette mismatch with manual calculation"
