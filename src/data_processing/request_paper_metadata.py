import json
import os

import pandas as pd
import requests
from tqdm import tqdm

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, "..", "..", "data")
request_fields = "externalIds,referenceCount,citationCount,title,url,year,authors,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,references"
semantic_url = "https://api.semanticscholar.org/graph/v1/paper/batch"
batch_size = 100


def choose_df():
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
    r = requests.post(
        semantic_url,
        params={"fields": request_fields},
        json={"ids": ids},
    )
    return r.json()


def save_papers_data(papers_data):
    out_path = os.path.join(data_dir, "papers_metadata.json")
    with open(out_path, "w") as f:
        json.dump(papers_data, f, indent=4)


def process_df(df):
    papers_data = []

    for i in tqdm(range(0, len(df), batch_size)):
        ids = df["id"].iloc[i : i + batch_size].tolist()
        ids = [f"ARXIV:{id}" for id in ids]
        results = request_papers(ids)
        papers_data.extend(results)
        break

    # build a pandas dataframe from the data (every papermetadata is a returned json)
    papers_data = pd.DataFrame(papers_data)
    df_path = os.path.join(data_dir, "papers_metadata.csv")
    papers_data.to_csv(df_path, index=False)

    return papers_data


def main():
    paper_df = choose_df()
    papers_data = process_df(paper_df)
    save_papers_data(papers_data)


if __name__ == "__main__":
    main()
