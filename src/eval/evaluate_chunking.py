#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib.pyplot as plt
import pandas as pd
import ragas
from datasets import Dataset, load_from_disk
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

import utils
import chains


def load_env_vars() -> tuple:
    """
    Load environment variables from a .env file.

    This function loads environment variables using the `load_dotenv` function
    and retrieves the values of `QDRANT_API_KEY` and `QDRANT_URL` from the
    environment.

    Returns:
        tuple: A tuple containing the `QDRANT_API_KEY` and `QDRANT_URL` values
               retrieved from the environment variables.
    """
    """Load environment variables from a .env file."""

    load_dotenv()
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    return qdrant_api_key, qdrant_url


def load_data(dataset_path: str) -> Dataset:
    """
    Load a dataset from disk.

    Args:
        dataset_path (str): The path to the dataset on disk.

    Returns:
        Dataset: The loaded dataset.
    """
    """Load a dataset from disk."""
    return load_from_disk(dataset_path)


def get_chain(chain_type: str, llm: object, retriever: object) -> object:
    """
    Get the chain based on the type.
    Parameters:
    chain_type (str): The type of chain to retrieve. Valid options are "stuff", "reduce", "rerank", and "hyde".
    llm: The language model to be used in the chain.
    retriever: The retriever to be used in the chain.
    Returns:
    chain: The chain object corresponding to the specified chain_type.
    Raises:
    ValueError: If an invalid chain_type is provided.
    """
    """Get the chain based on the type."""
    if chain_type == "stuff":
        chain = chains.stuff_chain(rag_llm=llm, rag_retriever=retriever)
    elif chain_type == "reduce":
        chain = chains.reduce_chain(qa_llm=llm, reduce_llm=llm, rag_retriever=retriever)
    elif chain_type == "rerank":
        chain = chains.reranker_chain(rag_llm=llm, rag_retriever=retriever)
    elif chain_type == "hyde":
        chain = chains.hyde_chain(rag_llm=llm, rag_retriever=retriever)
    else:
        raise ValueError(f"Invalid chain type: {chain_type}")

    return chain


def build_dataset(dataset: dict, chain: object, retriever: object) -> Dataset:
    """
    Perform retrieval-augmented generation (RAG) and evaluate the answers.
    Args:
        dataset (dict): A dictionary containing the dataset with keys "questions" and "ground_truth".
        chain (object): An object representing the language model chain used for generating answers.
        retriever (object): An object used to retrieve relevant documents based on the questions.
    Returns:
        Dataset: A dataset object containing the questions, generated answers, contexts, and ground truth answers.
    """
    """Perform retrieval-augmented generation (RAG) and evaluate the answers."""

    questions = dataset["questions"]
    ground_truth = dataset["ground_truth"]
    if isinstance(dataset[0], list):
        ground_truth = [gt[0] for gt in ground_truth]

    data = {"question": [], "answer": [], "contexts": [], "ground_truth": ground_truth}

    # Simulate RAG process: querying and retrieving documents
    for query in questions:
        docs = retriever.invoke(query, top_k=4)  # Retrieve top-k documents
        answer = chain.invoke({"question": query, "input_documents": docs})  # Get LLM answer
        context = [doc.page_content for doc in docs]  # Extract context

        # Append results
        data["question"].append(query)
        data["answer"].append(answer["output_text"])
        data["contexts"].append(context)

    data = Dataset.from_dict(data)

    return data


def evaluation(dataset: Dataset) -> pd.DataFrame:
    """
    Evaluate the dataset using RAGAS metrics.
    Parameters:
    dataset (Dataset): The dataset to be evaluated.
    Returns:
    pandas.DataFrame: The evaluation results as a pandas DataFrame.
    """
    """Evaluate the dataset using RAGAS metrics."""
    result = ragas.evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    )

    return result.to_pandas()


def main(dataset_path: str, chain_type: str) -> None:
    """
    Main function to handle loading data, processing, and evaluation.
    Args:
        dataset_path (str): The path to the dataset to be evaluated.
        chain_type (str): The type of chain to be used for processing.
    Returns:
        None
    """
    """Main function to handle loading data, processing, and evaluation."""

    # Initialize
    qdrant_api_key, qdrant_url = load_env_vars()

    vs = utils.load_vectorstore(qdrant_url, qdrant_api_key)
    retriever = vs.as_retriever()

    llm = utils.load_llm()
    chain = get_chain(chain_type, llm, retriever)

    # Dataset
    ds = load_data(dataset_path)
    data = build_dataset(ds, chain, retriever)

    # Evaluate
    evaluation_results = evaluation(data)

    # Show results
    print(evaluation_results)


if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Run RAG-based question answering and evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset")
    parser.add_argument(
        "--chain_type", type=str, default="stuff", required=True, help="Type of chain to use (e.g., stuff)"
    )
    # parser.add_argument("--chunking_method", type=str, required=False, default=None, help="Chunking method (optional)")

    args = parser.parse_args()

    main(args.dataset, args.chain_type)
