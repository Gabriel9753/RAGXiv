"""
Initialize Chroma vectorstore.
"""

import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import pandas as pd
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
from db_manager import DBManager, init_db

# add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import IndexConfig
from data_processing.data_utils import load_data


load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def load_pdf(file):
    loader = PyPDFLoader(file)
    return loader.load()[0]


def index(cfg, papers_df, paper_authors, paper_references):
    vectorstore_path = cfg.vectorstore_path
    embedding_model_name = cfg.embedding_model_name
    text_splitter_args = cfg.text_splitter_args
    device = cfg.device

    pdfs = papers_df["pdf_path"].tolist()
    if cfg.limit:
        pdfs = pdfs[: cfg.limit]

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": device})

    print(f"Indexing {len(pdfs)} PDFs.")

    # 1. Initialize database manager
    init_db()
    db_manager = DBManager()

    # 2. Extract data using multithreading
    documents = []
    with tqdm(total=len(pdfs), desc="Extracting text from PDFs") as pbar:
        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {executor.submit(load_pdf, file): file for file in pdfs}
            for future in as_completed(futures):
                try:
                    doc = future.result()
                    documents.append(doc)
                except Exception as e:
                    print(f"Error loading {futures[future]}: {e}")
                finally:
                    pbar.update(1)

    text_splitter = RecursiveCharacterTextSplitter(**text_splitter_args)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(chunks)} documents into chunks.")

    added_arxiv_ids = set()

    # 3. Insert metadata into chunks and save to PostgreSQL
    for chunk in chunks:
        source = chunk.metadata["source"]
        arxiv_id = os.path.basename(source).replace(".pdf", "")
        chunk.metadata["arxiv_id"] = arxiv_id
        added_arxiv_ids.add(arxiv_id)

    # 4. Insert metadata into PostgreSQL
    for index, row in papers_df.iterrows():
        arxiv_id = row["arxiv_id"]
        semantic_scholar_id = row["semantic_scholar_id"]
        if arxiv_id not in added_arxiv_ids:
            continue

        # Get authors and references
        authors = paper_authors.get(arxiv_id, [])
        references = paper_references.get(arxiv_id, [])

        # Insert into PostgreSQL using the DBManager
        db_manager.insert_paper(
            arxiv_id=str(arxiv_id),
            semantic_scholar_id=str(semantic_scholar_id),
            title=str(row["title"]),
            super_category=str(row["super_category"]),
            update_year=int(row["update_year"]),
            reference_count=int(row["reference_count"]),
            citation_count=int(row["citation_count"]),
            author_count=int(row["author_count"]),
            authors=authors,
            references=references
        )

    # 5. Save to vectorstore
    url = "https://1ed4f85b-722b-4080-97a7-afe8eab7ae7a.europe-west3-0.gcp.cloud.qdrant.io:6333"
    client = QdrantClient(url=url, api_key=QDRANT_API_KEY)
    if client.collection_exists(cfg.collection_name):
        client.delete_collection(cfg.collection_name)

    client.create_collection(
        collection_name=cfg.collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        embedding=embeddings,
        collection_name=cfg.collection_name,
    )

    vectorstore.from_documents(
        chunks, embedding=embeddings, url=url, api_key=QDRANT_API_KEY, collection_name=cfg.collection_name
    )
    db_manager.close()



def main():
    cfg = IndexConfig()
    papers_df, paper_authors, paper_references = load_data(drop_missing=cfg.drop_missing)
    index(cfg, papers_df, paper_authors, paper_references)


if __name__ == "__main__":
    main()
