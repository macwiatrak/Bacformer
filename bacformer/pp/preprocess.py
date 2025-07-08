import gzip
from typing import Any

import pandas as pd
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


def preprocess_genome_assembly(filepath: str) -> dict[str, Any]:
    """Preprocess a genome assembly file (GenBank or GFF) to extract protein information for Bacformer input.

    Args:
        filepath (str): Path to the genome assembly file (.gbff, .gbff.gz, .gff, or .gff.gz).

    Returns
    -------
        pd.DataFrame: A DataFrame containing protein information.
    """
    if filepath.endswith((".gbff", ".gbff.gz")):
        df = extract_protein_info_from_genbank(filepath)
    elif filepath.endswith((".gff", ".gff.gz")):
        df = extract_protein_info_from_gff(filepath)
    else:
        raise ValueError("Unsupported file format. Use .gbff, .gbff.gz, .gff, or .gff.gz.")

    # groupby contig and aggregate protein information
    df = (
        df.groupby(["strain_name", "accession_id", "accession_name", "contig_idx"])[
            ["gene_name", "protein_name", "start", "end", "protein_id", "protein_sequence"]
        ]
        .agg(list)
        .reset_index()
    )

    # sort by contig_idx
    df = df.sort_values(by="contig_idx", ascending=True)

    # aggregate all contigs in the genome
    df = (
        df.groupby(["strain_name"])[
            [
                "accession_id",
                "accession_name",
                "contig_idx",
                "gene_name",
                "protein_name",
                "start",
                "end",
                "protein_id",
                "protein_sequence",
            ]
        ]
        .agg(list)
        .reset_index()
    )
    return dict(df.iloc[0])
