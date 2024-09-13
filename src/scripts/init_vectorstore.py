"""
Initialize Chroma vectorstore.
"""

import os
import shutil

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader


def index(datapath, text_splitter_args: dict, chroma_path):
    # 1. Load and extract data
    loader = PyPDFLoader(datapath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(**text_splitter_args)
    chunks = text_splitter.split_documents(documents)
    # 5. Save to vectorstore
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
    # embeddings = Embeddings() # TODO: INSERT EMBDDINGS
    db = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_path)
    print(f"Indexed {len(chunks)} chunks to {chroma_path}")


if __name__ == "__main__":
    # TODO: ADD ARGPARSE
    index(DATAPATH, TEXT_SPLITTER_ARGS, CHROMA_PATH)
