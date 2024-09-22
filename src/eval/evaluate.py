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


def load_env_vars():
    """Load environment variables from a .env file."""
    load_dotenv()
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    return qdrant_api_key, qdrant_url


def load_data(dataset_path):
    """Load a dataset from disk."""
    return load_from_disk(dataset_path)


def get_chain(chain_type, llm, retriever):
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


def build_dataset(dataset, chain, retriever):
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


def evaluation(dataset, llm):
    """Evaluate the dataset using RAGAS metrics."""
    result = ragas.evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=llm,
    )

    return result.to_pandas()


def main(dataset_path, chain_type):
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
    evaluation_results = evaluation(data, llm)

    # Show results
    print(evaluation_results)


if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Run RAG-based question answering and evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset")
    parser.add_argument(
        "--chain_type", type=str, default="stuff", required=True, help="Type of chain to use (e.g., stuff)"
    )
    parser.add_argument("--chunking_method", type=str, required=False, default=None, help="Chunking method (optional)")

    args = parser.parse_args()

    main(args.dataset, args.chain_type, args.chunking_method)
