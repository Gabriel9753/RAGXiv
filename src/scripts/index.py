"""
Initialize Chroma vectorstore.
"""

import os
import shutil

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings



def index(datapath, text_splitter_args: dict, chroma_path, embedding_model_name):
    # 1. Load data
    pdfs = []
    for root, dirs, files in os.walk(datapath):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, file))
                if len(pdfs) > 10:
                    break

    pdfs = pdfs[:10] #TODO: Only for testing

    print(f"Found {len(pdfs)} PDF files in {datapath}.")
    # 2. Extract data
    documents = []
    for file in pdfs:
        loader = PyPDFLoader(file)
        doc = loader.load()[0]
        documents.append(doc)
        print(f"---- Document: {doc.metadata['source']} ----")
        print(doc.page_content[:1000])
    print(f"Extracted {len(documents)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(**text_splitter_args)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(chunks)} documents into chunks.")


    # 5. Save to vectorstore
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_path)
    print(f"Indexed {len(chunks)} chunks to {chroma_path}")


if __name__ == "__main__":
    # TODO: ADD ARGPARSE
    DATAPATH = r"D:\Personal\OneDrive\Gabrilyi\summerschool_project\papers"
    CHROMAPATH = r"./chroma"
    EMBEDDING_MODEL_NAME = "sentence-transformers/allenai-specter"
    TEXT_SPLITTER_ARGS = {
        "chunk_size": 300,
        "chunk_overlap": 100,
        "length_function": len,
        "add_start_index": True,
    }

    index(DATAPATH, TEXT_SPLITTER_ARGS, CHROMAPATH, EMBEDDING_MODEL_NAME)
