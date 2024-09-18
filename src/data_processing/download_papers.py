import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PyPDF2 import PdfReader
from requests.exceptions import HTTPError, RequestException
from retry import retry
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import Config

cfg = Config()

params = {"switchLocale": "y", "siteEntryPassthrough": "true"}


@retry(tries=3, delay=1, exceptions=(RequestException, HTTPError))
def download_paper(row, url_template, paper_dir):
    id = row["id"]
    url = url_template.format(id=id)
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raises an exception for non-200 status codes
    file_path = os.path.join(paper_dir, f"{id}.pdf")
    with open(file_path, "wb") as f:
        f.write(response.content)
    return id


def checkFile(fullfile):
    with open(fullfile, "rb") as f:
        try:
            pdf = PdfReader(f)
            info = pdf.metadata
            if info:
                return True
            else:
                return False
        except Exception as e:
            return False


def verify_files(ids, paper_dir):
    files = os.listdir(paper_dir)
    corrupted_files = []
    with tqdm(total=len(files), desc="Verifying files") as pbar:
        for file in files:
            if file.endswith(".pdf"):
                id = os.path.basename(file).replace(".pdf", "")
                if id not in ids:
                    corrupted_files.append(file)
                else:
                    if not checkFile(os.path.join(paper_dir, file)):
                        corrupted_files.append(file)
            pbar.update(1)
            pbar.set_postfix({"corrupted": len(corrupted_files)})
    corrupted_ids = [os.path.basename(file).replace(".pdf", "") for file in corrupted_files]
    return corrupted_files, corrupted_ids


def get_missing_files(ids, paper_dir):
    files = os.listdir(paper_dir)
    missing_files = []
    for id in ids:
        if f"{id}.pdf" not in files:
            missing_files.append(id)
    return missing_files


def main():
    cfg = Config()
    csv_path = cfg.datapath
    paper_dir = cfg.paper_dir
    multi_thread_workers = cfg.multi_thread_workers

    paper_df = pd.read_csv(csv_path)
    # paper_df = paper_df.sample(4).reset_index(drop=True)  # For testing purposes

    # url_template = "https://arxiv.org/pdf/{id}"
    url_template = "https://export.arxiv.org/pdf/{id}"

    # Create the paper_dir if it doesn't exist
    os.makedirs(paper_dir, exist_ok=True)

    # Use ThreadPoolExecutor for concurrent downloads
    with ThreadPoolExecutor(max_workers=multi_thread_workers) as executor:
        futures = []
        for _, row in paper_df.iterrows():
            future = executor.submit(download_paper, row, url_template, paper_dir)
            futures.append(future)

        # Use tqdm to show progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading papers"):
            pass

    corrupted_files, corrupted_ids = verify_files(paper_df["id"].astype(str).tolist(), paper_dir)

    # Remove corrupted files
    for file in corrupted_files:
        os.remove(os.path.join(paper_dir, file))

    print(f"Removed {len(corrupted_files)} corrupted files.")
    print(f"Before: {len(paper_df)} papers")
    paper_df = paper_df[~paper_df["id"].astype(str).isin(corrupted_ids)]

    missing_ids = get_missing_files(paper_df["id"].astype(str).tolist(), paper_dir)
    print(f"Missing {len(missing_ids)} files.")
    paper_df = paper_df[~paper_df["id"].astype(str).isin(missing_ids)]

    paper_df.to_csv(cfg.datapath, index=False)
    print(f"After: {len(paper_df)} papers")


if __name__ == "__main__":
    main()
