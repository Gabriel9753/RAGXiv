import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, "..", "data")

csv_file = os.path.join(data_dir, "sample_2024.csv")
paper_dir = os.path.join(data_dir, "papers")
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
    paper_df = pd.read_csv(csv_file)

    url_template = "https://arxiv.org/pdf/{id}"

    # Create the paper_dir if it doesn't exist
    os.makedirs(paper_dir, exist_ok=True)

    # Use ThreadPoolExecutor for concurrent downloads
    with ThreadPoolExecutor(max_workers=10) as executor:
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
