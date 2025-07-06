from bacformer.pp.embed_prot_seqs import embed_dataset_col_with_bacformer
from bacformer.tl import (
    get_intergenic_bp_dist,
    operon_prot_indices_to_pairwise_labels,
    predict_pairwise_operon_boundaries,
)
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    # read the data from HuggingFace
    # load the dataset
    dataset = load_dataset("macwiatrak/operon-identification-long-read-rna-sequencing", split="test")

    # embed the protein sequences with Bacformer and convert to pandas
    df = embed_dataset_col_with_bacformer(
        dataset=dataset,
        protein_sequences_col="protein_sequence",
        bacformer_model_path="macwiatrak/bacformer-masked-complete-genomes",
        max_n_proteins=9000,
        genome_pooling_method="mean",
    ).to_pandas()

    # explode the dataset by contig, this allows prediction per contig
    df = df.explode(
        [
            "contig_name",
            "gene_name",
            "locus_tag",
            "start",
            "end",
            "strand",
            "protein_id",
            "embeddings",
            "operon_prot_indices",
        ]
    )

    # compute the intergenic distances
    df["intergenic_bp"] = df.apply(
        lambda row: get_intergenic_bp_dist(
            starts=row["start"],
            ends=row["end"],
        ),
        axis=1,
    )

    # run the operon prediction
    df["operon_pairwise_scores"] = df.apply(
        lambda row: predict_pairwise_operon_boundaries(
            emb=row["embeddings"],
            intergenic_bp=row["intergenic_bp"],
            strand=row["strand"],
            scale_bp=500,
            max_gap=500,
        ),
        axis=1,
    )

    # get the labels
    df["operon_pairwise_labels"] = df.apply(
        lambda row: operon_prot_indices_to_pairwise_labels(
            operon_prot_indices=row["operon_prot_indices"],
            n_genes=len(row["protein_sequence"]),
        ),
        axis=1,
    )

    # compute auroc
    df["auroc"] = df.apply(
        lambda row: roc_auc_score(row["operon_pairwise_labels"], row["operon_pairwise_scores"]), axis=1
    )

    df.to_parquet(
        "operon_prediction_results.parquet",
    )

    # plot the AUROC curves
