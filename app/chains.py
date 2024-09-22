import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableBranch,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import utils
import templates


def stuff_chain(rag_llm:Runnable, rag_retriever:VectorStoreRetriever, with_guard=False):
    """Build a runnable with message history"""

    retrieve_documents = RunnableBranch(
    (
        # Both empty string and empty list evaluate to False
        lambda x: not x.get("chat_history", False),
        # If no chat history, then we just pass input to retriever
        (lambda x: x["input"]) | rag_retriever,
    ),
    # If chat history, then we pass inputs to LLM chain, then to retriever
        templates.CONTEXTUALIZE_PROMPT | rag_llm | StrOutputParser() | rag_retriever)


    # Stuff documents chain
    # "Stuffs" or inputs the documents with the final prompt for the LLM
    question_answer_chain = (
        RunnablePassthrough.assign(context=utils.format_docs)
        | templates.QA_PROMPT
        | rag_llm
        | StrOutputParser()
    )

    # RAG chain
    # Combines the history-aware retriever and the question-answer chain
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=retrieve_documents
        ).assign(answer=question_answer_chain)
    )

    return retrieval_chain




def reduce_chain(qa_llm:Runnable, reduce_llm:Runnable,  rag_retriever:VectorStoreRetriever):
    """Build Reduce chain.

    Args:
        qa_llm (Runnable): The LLM for question answering.
        reduce_llm (Runnable): The LLM for reducing documents.
        rag_retriever (VectorStoreRetriever): The retriever for RAG.
        rag_memory (Memory): The memory for RAG.
        keys (dict, optional): The keys for the input, history, and output messages. Defaults to None.

    Returns:
        RunnableWithMessageHistory: The runnable with message history.
    """

    # Retrieve
    # Contextualizes the user input based on the chat history for the retrieval
    retrieve_documents= RunnableBranch( #: Runnable[Any, list[Document]]
    (
        # Both empty string and empty list evaluate to False
        lambda x: not x.get("chat_history", False),
        # If no chat history, then we just pass input to retriever
        (lambda x: x["input"]) | rag_retriever,
    ),
    # If chat history, then we pass inputs to LLM chain, then to retriever
        templates.CONTEXTUALIZE_PROMPT | reduce_llm | StrOutputParser() | rag_retriever)

    # Collapse documents chain
    # If max_tokens is exceeded, collapse the document string through summarization with llm
    collapse_chain = (
        {"input": lambda x: x["input"], "chat_history": lambda x: x["chat_history"]}
        | RunnablePassthrough.assign(
            context=retrieve_documents | utils.format_docs # Combine documents
        )
        | templates.COLLAPSE_PROMPT
        | reduce_llm
        | StrOutputParser()
    )

    # Question-Answer chain
    qa_chain = (
        {"context": collapse_chain,
         "input": lambda x: x["input"],
         "chat_history": lambda x: x["chat_history"]}
        | templates.QA_PROMPT
        | qa_llm
        | StrOutputParser()
    )


    # # RAG chain
    # # Combines the history-aware retriever and the question-answer chain
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=collapse_chain
        ).assign(answer=qa_chain)
    )
    return retrieval_chain

def reranker(rag_retriever):
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=rag_retriever
    )
    return compression_retriever

def reranker_chain(rag_llm, rag_retriever):

    reranking_retriever = reranker(rag_retriever)
    retrieve_documents = RunnableBranch(
    (
        # Both empty string and empty list evaluate to False
        lambda x: not x.get("chat_history", False),
        # If no chat history, then we just pass input to retriever
        (lambda x: x["input"]) | reranking_retriever,
    ),
    # If chat history, then we pass inputs to LLM chain, then to retriever
        templates.CONTEXTUALIZE_PROMPT | rag_llm | StrOutputParser() | reranking_retriever)


    # Stuff documents chain
    # "Stuffs" or inputs the documents with the final prompt for the LLM
    question_answer_chain = (
        RunnablePassthrough.assign(context=utils.format_docs)
        | templates.QA_PROMPT
        | rag_llm
        | StrOutputParser()
    )

    # RAG chain
    # Combines the history-aware retriever and the question-answer chain
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=retrieve_documents
        ).assign(answer=question_answer_chain)
    )

    return retrieval_chain


def hyde_chain(rag_llm:Runnable, rag_retriever:VectorStoreRetriever):
    """Build Hypothetical Document Embedding chain.

    Args:
        rag_llm (Runnable): The LLM for RAG.
        rag_retriever (VectorStoreRetriever): The retriever for RAG.

    Returns:
        Runnable: The runnable for the hypothetical document embedding.
    """

    retrieve_documents = RunnableBranch(
    (
        lambda x: not x.get("chat_history", False),
        (lambda x: x["input"]) | rag_retriever,
    ),
        templates.HYDE_PROMPT | rag_llm | StrOutputParser() | rag_retriever)


    question_answer_chain = (
        RunnablePassthrough.assign(context=utils.format_docs)
        | templates.QA_PROMPT
        | rag_llm
        | StrOutputParser()
    )

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=retrieve_documents
        ).assign(answer=question_answer_chain)
    )

    return retrieval_chain




def semantic_search_chain(rag_llm:Runnable, rag_retriever:VectorStoreRetriever):
    """Build Semantic Search chain.

    Args:
        query (str): The query for the semantic search.
        rag_llm (Runnable): The LLM for RAG.
        rag_retriever (VectorStoreRetriever): The retriever for RAG.

    Returns:
"""

    llm_chain = (
        RunnablePassthrough.assign(context=(lambda x: utils.format_docs(x["context"])))
        | templates.SEMANTIC_SEARCH_PROMPT
        | rag_llm
        | StrOutputParser()
    )

    chain = RunnableParallel(
        {"context": rag_retriever, "input": RunnablePassthrough()}
    ).assign(answer=llm_chain)

    return chain


def paper_qa_chain(rag_llm:Runnable):
    # Not fixed yet

    llm = utils.load_llm(temp=0.3)
    prompt = templates.PAPERQA_PROMPT
    chain = (prompt | llm)

    return chain


def summarization_chain(rag_llm):
    prompt = templates.SUMMARIZATION_PROMPT
    chain = (prompt | rag_llm | StrOutputParser())

    return chain