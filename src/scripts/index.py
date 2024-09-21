"""
Initialize Chroma vectorstore.
"""

import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
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

TEXT_SPLITTER = {
    "SemanticChunker": SemanticChunker,
    "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
}

def load_pdf(file):
    loader = PyPDFLoader(file)
    return loader.load()


def index(cfg, papers_df, paper_authors, paper_references):
    embedding_model_name = cfg.embedding_model_name
    text_splitter_args = cfg.text_splitter_args
    device = cfg.device

    pdfs = papers_df["pdf_path"].tolist()
    if cfg.limit:
        pdfs = pdfs[: cfg.limit]

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": device})

    print(f"Indexing {len(pdfs)} PDFs.")

    # 1. Initialize database manager
    init_db(clear=True)
    db_manager = DBManager()

    # 2. Extract data using multithreading
    documents = []
    with tqdm(total=len(pdfs), desc="Extracting text from PDFs") as pbar:
        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {executor.submit(load_pdf, file): file for file in pdfs}
            for future in as_completed(futures):
                try:
                    doc = future.result()
                    documents.extend(doc)
                except Exception as e:
                    print(f"Error loading {futures[future]}: {e}")
                finally:
                    pbar.update(1)

    print(f"Amount of documents: {len(documents)}")
    # change all the content text to utf-8
    for doc in documents:
        doc.page_content = doc.page_content.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    # traditional_text_splitter = RecursiveCharacterTextSplitter(**text_splitter_args)
    # traditional_chunks = traditional_text_splitter.split_documents(documents)
    # print(f"Split {len(traditional_chunks)} documents into chunks.")

    # semantic_text_splitter = SemanticChunker(embeddings=embeddings, add_start_index=True)
    # semantic_chunks = semantic_text_splitter.split_documents(documents)
    # print(f"Split {len(semantic_chunks)} documents into chunks.")

    for arg in text_splitter_args:
        text_splitter_type = arg["type"]
        arg.pop("type")
        try:
            text_splitter = TEXT_SPLITTER[text_splitter_type](**arg)
        except:
            text_splitter = TEXT_SPLITTER[text_splitter_type](**arg, embeddings=embeddings)
        chunks = text_splitter.split_documents(documents)

        print(f"{text_splitter_type}: Split in {len(chunks)} chunks.")
        added_arxiv_ids = set()
        # 3. Insert metadata into chunks and save to PostgreSQL
        for chunk in chunks:
            source = chunk.metadata["source"]
            arxiv_id = os.path.basename(source).replace(".pdf", "")
            chunk.metadata["arxiv_id"] = arxiv_id
            added_arxiv_ids.add(arxiv_id)

        if text_splitter_type == cfg.database_for:
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
        client = QdrantClient(url=cfg.qdrant_url, api_key=QDRANT_API_KEY)
        # local client
        # client = QdrantClient(url="http://localhost:6333", api_key=QDRANT_API_KEY)
        collection_name = f"{cfg.collection_name}_{text_splitter_type}"
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

        vectorstore = QdrantVectorStore(
            client=client,
            embedding=embeddings,
            collection_name=collection_name,
        )

        vectorstore.from_documents(
            chunks, embedding=embeddings, url=cfg.qdrant_url, api_key=QDRANT_API_KEY, collection_name=collection_name
        )

    return

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
