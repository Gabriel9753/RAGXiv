#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from dotenv import load_dotenv
from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
import config
import utils

def load_env_vars():
    """
    Load required environment variables from the .env file.

    Returns:
        tuple: A tuple containing Qdrant API key and URL.

    Raises:
        EnvironmentError: If the required environment variables are not found.
    """
    load_dotenv()  # Load environment variables from a .env file
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")

    # Raise an error if any required environment variable is missing
    if not qdrant_api_key or not qdrant_url:
        raise EnvironmentError("Missing QDRANT_API_KEY or QDRANT_URL in environment.")

    return qdrant_api_key, qdrant_url


def main():
    """
    Main function to handle the indexing of documents into the vector store.

    This function loads environment variables, initializes the vector store, and
    uses the SQLRecordManager to index documents. The indexing process associates
    source IDs with the corresponding documents.

    Steps:
        1. Load environment variables.
        2. Initialize the vector store.
        3. Configure the SQL record manager.
        4. Index documents into the vector store.
    """

    # Load environment variables
    qdrant_api_key, qdrant_url = load_env_vars()

    # Load vector store using custom utility function
    vs = utils.load_vectorstore(qdrant_url, qdrant_api_key)

    # Get the collection namespace from the config
    namespace = config.COLLECTION_NAME

    # Initialize SQL record manager to manage records in the vector store
    record_manager = SQLRecordManager(
        namespace=namespace,  # Define namespace for the collection
        db_url=qdrant_url  # Use Qdrant URL for database interaction
    )

    # Index documents into the vector store
    index(
        vectorstore=vs,            # Vector store instance
        record_manager=record_manager,  # SQL-based record manager
        namespace=namespace,        # Namespace for document indexing
        source_id_key="source"      # Key used to identify the document source
    )


if __name__ == "__main__":
    main()
