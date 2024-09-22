import json
import os
import sys

import pandas as pd
import requests
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import Config

cfg = Config()

data_dir = cfg.data_dir
paper_metadata_path = cfg.paper_metadata_path

request_fields = "externalIds,referenceCount,citationCount,title,url,year,authors,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,references"
semantic_url = "https://api.semanticscholar.org/graph/v1/paper/batch"
batch_size = 50


def choose_df():
    """
    Prompts the user to choose a CSV file from the data directory and loads it into a DataFrame.

    This function lists all CSV files in the data directory, allows the user to choose one by
    entering a number, and then loads the chosen file into a pandas DataFrame.

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the chosen CSV file.

    Raises:
        Exception: If there's an error in file selection or loading.
    """
    all_csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    print("Choose a CSV file by typing the number:")
    for i, f in enumerate(all_csv_files):
        print(f"➜    ({i:02d}) {f}")
    try:
        choice = int(input("Enter the number: "))
        csv_file = os.path.join(data_dir, all_csv_files[choice])
        paper_df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error: {e}")
        print("Please enter a valid number.")
        return choose_df()
    return paper_df


def request_papers(ids):
    """
    Sends a request to the Semantic Scholar API to retrieve paper metadata.

    Args:
        ids (list): A list of paper IDs to request metadata for.

    Returns:
        dict: The JSON response from the Semantic Scholar API containing paper metadata.
    """
    r = requests.post(
        semantic_url,
        params={"fields": request_fields},
        json={"ids": ids},
    )
    return r.json()


def save_papers_data(papers_data):
    """
    Saves the retrieved paper metadata to a JSON file.

    Args:
        papers_data (list): A list of dictionaries containing paper metadata.
    """
    with open(paper_metadata_path, "w") as f:
        json.dump(papers_data, f, indent=4)


def process_df(df):
    """
    Processes a DataFrame of papers by requesting metadata from Semantic Scholar in batches.

    This function iterates over the DataFrame in batches, requests metadata for each batch
    from the Semantic Scholar API, and collects all the results.

    Args:
        df (pandas.DataFrame): The DataFrame containing paper information.

    Returns:
        list: A list of dictionaries, each containing metadata for a single paper.
    """
    papers_data = []

    for i in tqdm(range(0, len(df), batch_size)):
        ids = df["id"].iloc[i : i + batch_size].tolist()
        ids = [f"ARXIV:{id}" for id in ids]
        results = request_papers(ids)
        papers_data.extend(results)

    return papers_data


def main():
    """
    Main function to orchestrate the paper metadata retrieval process.

    This function performs the following steps:
    1. Prompts the user to choose a CSV file with paper information.
    2. Processes the chosen DataFrame to retrieve metadata from Semantic Scholar.
    3. Saves the retrieved metadata to a JSON file.
    """
    paper_df = choose_df()
    papers_data = process_df(paper_df)
    save_papers_data(papers_data)


if __name__ == "__main__":
    main()
