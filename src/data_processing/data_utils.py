import json
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import Config

cfg = Config()


def load_data(drop_missing=False):
    """
    Load and process paper data from CSV and JSON files.

    This function reads paper data from a CSV file and additional metadata from a JSON file,
    merges them, and performs some data preprocessing.

    Args:
        drop_missing (bool, optional): If True, drops papers that are not in the additional data.
                                       Defaults to False.

    Returns:
        tuple: A tuple containing three elements:
            - papers_df (pandas.DataFrame): A DataFrame containing merged and processed paper data.
            - paper_authors (dict): A dictionary mapping arXiv IDs to lists of author information.
            - paper_references (dict): A dictionary mapping arXiv IDs to lists of reference information.

    Note:
        - This function assumes the existence of a Config class with specific attributes.
        - It reads data from files specified in the Config instance.
        - The function performs various data processing steps, including:
          * Converting arXiv IDs to strings
          * Creating PDF file paths
          * Extracting additional metadata from the JSON file
          * Merging CSV data with additional metadata
    """
    papers_df = pd.read_csv(cfg.datapath)
    with open(cfg.paper_metadata_path) as f:
        papers_data = json.load(f)

    papers_df["arxiv_id"] = papers_df["id"].astype(str)
    papers_df["pdf_path"] = papers_df["arxiv_id"].apply(lambda x: os.path.join(cfg.paper_dir, f"{x}.pdf"))

    additional_data = defaultdict(list)
    paper_authors = dict()
    paper_references = dict()

    for i, paper in enumerate(papers_data):
        if paper is None:
            continue
        semantic_scholar_id = paper["paperId"]
        arxiv_id = paper["externalIds"]["ArXiv"]
        doi = paper["externalIds"].get("DOI", None)
        reference_count = paper.get("referenceCount", None)
        citation_count = paper.get("citationCount", None)
        publication_type = paper.get("publicationTypes", None)
        publication_type = publication_type[0] if publication_type is not None else None
        paper_authors[arxiv_id] = paper["authors"]
        author_count = len(paper_authors[arxiv_id])
        paper_references[arxiv_id] = paper["references"]

        additional_data["arxiv_id"].append(arxiv_id)
        additional_data["doi"].append(doi)
        additional_data["reference_count"].append(reference_count)
        additional_data["citation_count"].append(citation_count)
        additional_data["publication_type"].append(publication_type)
        additional_data["semantic_scholar_id"].append(semantic_scholar_id)
        additional_data["author_count"].append(author_count)

    additional_data = pd.DataFrame(additional_data)

    if drop_missing:
        # drop continued papers (papers that are not in the additional data)
        papers_df = papers_df[papers_df["arxiv_id"].isin(additional_data["arxiv_id"])]

    papers_df = pd.merge(papers_df, additional_data, on="arxiv_id", how="left")

    return papers_df, paper_authors, paper_references
