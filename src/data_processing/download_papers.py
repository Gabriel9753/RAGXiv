import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from configs import Config
from tqdm import tqdm

cfg = Config()

os.makedirs(paper_dir, exist_ok=True)


def download_paper(row, url_template, paper_dir):
    id = row["id"]
    url = url_template.format(id=id)
    response = requests.get(url)
    file_path = os.path.join(paper_dir, f"{id}.pdf")
    with open(file_path, "wb") as f:
        f.write(response.content)
    return id


def main():
    cfg = Config()
    csv_path = cfg.datapath
    paper_dir = cfg.paper_dir
    multi_thread_workers = cfg.multi_thread_workers

    paper_df = pd.read_csv(csv_path)

    url_template = "https://arxiv.org/pdf/{id}"

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
            id = future.result()
            # You can add additional processing here if needed


if __name__ == "__main__":
    main()
