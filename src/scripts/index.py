"""
Initialize Chroma vectorstore.
"""

import os
import shutil
import sys

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

# add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import IndexConfig
from data_processing.data_utils import load_data


def index(cfg, papers_df, paper_authors, paper_references):
    vectorstore_path = cfg.vectorstore_path
    embedding_model_name = cfg.embedding_model_name
    text_splitter_args = cfg.text_splitter_args
    device = cfg.device

    pdfs = papers_df["pdf_path"].tolist()
    if cfg.limit:
        pdfs = pdfs[:cfg.limit]

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": device})

    print(f"Indexing {len(pdfs)} PDFs.")

    # 2. Extract data
    documents = []
    with tqdm(total=len(pdfs), desc="Extracting text from PDFs") as pbar:
        for file in pdfs:
            loader = PyPDFLoader(file)
            doc = loader.load()[0]
            documents.append(doc)
            pbar.update(1)
    text_splitter = RecursiveCharacterTextSplitter(**text_splitter_args)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(chunks)} documents into chunks.")

    # 3. Insert metadata into chunks
    for chunk in chunks:
        source = chunk.metadata["source"]
        arxiv_id = os.path.basename(source).replace(".pdf", "")
        row = papers_df[papers_df["arxiv_id"] == arxiv_id].iloc[0]
        chunk.metadata["arxiv_id"] = arxiv_id
        chunk.metadata["title"] = str(row["title"])
        chunk.metadata["super_category"] = str(row["super_category"])
        chunk.metadata["reference_count"] = str(row["reference_count"])
        chunk.metadata["citation_count"] = str(row["citation_count"])
        chunk.metadata["author_count"] = str(row["author_count"])
        # TODO: Add metadata to another db for later use so not every chunk has the data...

    # 5. Save to vectorstore
    if os.path.exists(vectorstore_path):
        shutil.rmtree(vectorstore_path)

    client = QdrantClient(path=vectorstore_path)
    if client.collection_exists("arxiv_demo"):
        client.delete_collection("arxiv_demo")

    client.create_collection(
            collection_name="arxiv_demo",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="arxiv_demo",
        embedding=embeddings,
    ).from_documents(chunks, embedding=embeddings)

    print(f"Indexed {len(chunks)} chunks to {vectorstore_path}")


def main():
    cfg = IndexConfig()
    papers_df, paper_authors, paper_references = load_data(drop_missing=cfg.drop_missing)
    index(cfg, papers_df, paper_authors, paper_references)

if __name__ == "__main__":
    main()
