import anndata as ad
import numpy as np
import scanpy as sc
from bacformer.pp.embed_prot_seqs import embed_dataset_col_with_bacformer
from datasets import load_dataset
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import LabelEncoder


def run():
    """Run strain clustering example."""
    # load the dataset
    dataset = load_dataset("macwiatrak/strain-clustering-protein-sequences-sample", split="train")

    # embed the protein sequences with Bacformer
    dataset = embed_dataset_col_with_bacformer(
        dataset=dataset,
        protein_sequences_col="protein_sequence",
        bacformer_model_path="macwiatrak/bacformer-masked-MAG",
        max_n_proteins=9000,
        genome_pooling_method="mean",
    )

    # convert dataset to pandas DataFrame
    df = dataset.to_pandas()

    # create anndata object needed for clustering
    embeddings = np.stack(df["embeddings"].tolist())  # get embedding matrix
    adata = ad.AnnData(
        X=embeddings,
        obs=df.drop(columns=["embeddings"]).copy(),
    )

    # compute neighbors witg scanpy
    sc.pp.neighbors(adata, use_rep="X")

    # compute UMAP
    sc.tl.umap(adata)

    # plot UMAP by species
    sc.pl.umap(adata, color="species")

    # plot UMAP by family
    sc.pl.umap(adata, color="family")

    # compute clustering metrics (optional)
    sc.tl.leiden(adata, resolution=0.1, key_added="leiden_clusters")

    # Convert Leiden cluster labels to integer labels
    leiden_clusters = adata.obs["leiden_clusters"].astype(int)

    # Encode ground-truth labels
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(adata.obs["species"])

    # Compute ARI, NMI, and Silhouette
    ari = adjusted_rand_score(numeric_labels, leiden_clusters)
    nmi = normalized_mutual_info_score(numeric_labels, leiden_clusters)
    # Silhouette requires sample-level features + predicted labels
    sil = silhouette_score(adata.X, leiden_clusters)
    print(f"ARI: {ari:.3f}, NMI: {nmi:.3f}, Silhouette Score: {sil:.3f}")


if __name__ == "__main__":
    run()
