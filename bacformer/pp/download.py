import logging
import os
from typing import Literal
from xml.etree import ElementTree

import requests
from Bio import Entrez


def download_genome_assembly_by_taxid(
    taxid: int,
    file_type: Literal["gbff", "gff", "gtf", "fasta"] = "gbff",
    output_dir: str = None,
):
    """Search for assemblies by TaxID, pick the first (or best) assembly, then download the desired genomic data by file type.

    Args:
        taxid: int
            NCBI Taxonomy ID (e.g. 562 for E. coli K-12)
        file_type: str
            File type to download, one of "gbff", "gff", "gtf", "fasta"
        output_dir: str
            Output directory to save the downloaded file
    """
    if output_dir is None:
        output_dir = os.getcwd()

    # 1) E-search: find assembly UIDs
    esearch_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=assembly&term=txid{taxid}[Organism]&retmode=json"
    )
    resp = requests.get(esearch_url)
    resp.raise_for_status()
    data = resp.json()

    idlist = data["esearchresult"]["idlist"]
    if not idlist:
        logging.info(f"No assemblies found for TaxID {taxid}")
        return
    # Pick the first assembly UID, though you may want a more complex selection logic
    asm_uid = idlist[0]
    logging.info(f"Found assembly UID {asm_uid} for TaxID {taxid}")

    # 2) E-summary: get assembly metadata
    esummary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=assembly&id={asm_uid}&retmode=json"
    resp2 = requests.get(esummary_url)
    resp2.raise_for_status()
    data2 = resp2.json()

    docsum = data2["result"][asm_uid]
    # Get the GCF accession (RefSeq)
    gcf_accession = docsum["assemblyaccession"]  # e.g. "GCF_000123456.1"
    logging.info(f"GCF accession: {gcf_accession}")

    # RefSeq FTP path (also available: ftppath_genbank)
    refseq_ftp = docsum["ftppath_refseq"]  # e.g. "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/..."

    if not refseq_ftp:
        logging.info("No RefSeq FTP path found for this assembly.")
        return

    # 3) Construct the URL for the file
    # The last part of the ftp path is typically the same name as the file prefix
    # For example, if ftp_path is:
    #   ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/123/456/GCF_000123456.1_ASM12345v1
    # then the file prefix is "GCF_000123456.1_ASM12345v1"
    file_prefix = refseq_ftp.split("/")[-1]
    filename = file_prefix + f"_genomic.{file_type}.gz"

    # Use HTTPS instead of FTP to avoid potential FTP issues
    # Just replace "ftp://" with "https://", as NCBI supports both
    https_ftp = refseq_ftp.replace("ftp://", "https://")
    fasta_url = f"{https_ftp}/{filename}"
    logging.info(f"Downloading {fasta_url} ...")

    # 4) Download the file
    r = requests.get(fasta_url, stream=True)
    r.raise_for_status()
    out_file = os.path.join(output_dir, f"{taxid}_{gcf_accession}_genomic.{file_type}.gz")
    with open(out_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    logging.info(f"Downloaded genome assembly to {out_file}")


def download_refseq_assembly_entrez(
    assembly_id: str,
    file_type: Literal["gbff", "gff", "fasta"] = "gbff",
    email: str = "your_email@example.com",
    output_dir: str = "",
) -> str:
    """Download RefSeq assembly files (GenBank, FASTA, or GFF) from NCBI's Assembly database using Biopython Entrez calls to find the correct FTP/HTTP path.

    :param assembly_id:  RefSeq assembly accession (e.g. 'GCF_000006765.1').
    :param file_type:    One of: 'gbff' (GenBank), 'gff', 'fasta', or 'gtf'.
                        'gtf' is NOT directly provided by NCBI assembly downloads and
                        will raise an exception.
    :param email:        Your email address (required by NCBI).
    :param output_file:  Path/filename for the downloaded file (optional). If not specified,
                        we'll derive a name from the assembly and file type.

    :param output_dir:   Directory to save the downloaded file. If not specified, the current
                        working directory will be used.

    :return:            Path to the downloaded file.
    """
    # Validate file_type
    file_type = file_type.lower()
    valid_types = ["gbff", "gff", "fasta"]
    if file_type not in valid_types:
        raise ValueError(f"Invalid file_type '{file_type}'. Must be one of: {valid_types}")

    # Set up Entrez with your email
    Entrez.email = email

    # 1) ESearch in the 'assembly' database
    #    Query the assembly accession (e.g. GCF_000006765.1) to get the internal UID
    search_handle = Entrez.esearch(db="assembly", term=assembly_id, retmax=1)
    search_results = Entrez.read(search_handle)
    search_handle.close()

    id_list = search_results.get("IdList", [])
    if not id_list:
        raise ValueError(f"No assembly found in NCBI 'assembly' DB for '{assembly_id}'")

    uid = id_list[0]  # Should be exactly one result in most cases

    # 2) ESummary to get the assembly metadata, including the FTP link
    summary_handle = Entrez.esummary(db="assembly", id=uid, report="full")
    summary_xml = summary_handle.read()
    summary_handle.close()

    # 3) Parse the XML to extract the DocumentSummary and FtpPath_RefSeq (or FtpPath_GenBank if needed)
    #    We also need the 'AssemblyAccession' or 'AssemblyName' to build the final file path
    root = ElementTree.fromstring(summary_xml)
    docsum = root.find(".//DocumentSummary")
    if docsum is None:
        raise ValueError("Could not parse assembly DocumentSummary from ESummary output.")

    # FtpPath_RefSeq is usually: ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/.../GCF_000006765.1_ASMXXXvX
    ftp_path = docsum.findtext("FtpPath_RefSeq")
    # If for some reason there's no RefSeq FTP path, you might consider 'FtpPath_GenBank' instead.
    # ftp_path = docsum.findtext("FtpPath_GenBank")  # Alternative if no RefSeq path?

    if not ftp_path:
        raise ValueError(f"No FTP path found in ESummary for assembly '{assembly_id}'.")

    # 4) Also need the specific "assembly directory" name, e.g. GCF_000006765.1_ASM676v1
    #    Usually found in the last part of ftp_path
    #       ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/006/765/GCF_000006765.1_ASM676v1
    asm_name = os.path.basename(ftp_path)

    # 5) Decide the correct file suffix for each file_type
    #    NCBI typically uses:
    #      - FASTA => _genomic.fna.gz
    #      - GBFF  => _genomic.gbff.gz
    #      - GFF   => _genomic.gff.gz
    suffix_map = {"gbff": "_genomic.gbff.gz", "fasta": "_genomic.fna.gz", "gff": "_genomic.gff.gz"}
    suffix = suffix_map[file_type]

    # 6) Build the final download URL, replacing ftp:// with https:// for convenience
    #    e.g.: ftp_path + "/" + asm_name + "_genomic.gbff.gz"
    download_url = ftp_path.replace("ftp://", "https://") + "/" + asm_name + suffix

    # 7) Download with requests (streaming)
    r = requests.get(download_url, stream=True)
    r.raise_for_status()  # Raise an exception if there's a 4XX/5XX error

    output_file_path = os.path.join(output_dir, f"{assembly_id}_{asm_name}.{file_type}.gz")
    with open(output_file_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return output_file_path
