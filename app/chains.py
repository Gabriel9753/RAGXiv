import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
from langchain.output_parsers.regex import RegexParser
from langchain.chains.llm import LLMChain
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableBranch,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import utils
import templates


def stuff_chain(rag_llm:Runnable, rag_retriever:VectorStoreRetriever):
    """Build a runnable with message history"""

    # # History-aware retriever
    # # Contextualizes the user input based on the chat history for the retrieval
    # retrieve_documents = RunnableBranch(
    # (
    #     # Both empty string and empty list evaluate to False
    #     lambda x: not x.get("chat_history", False),
    #     # If no chat history, then we just pass input to retriever
    #     (lambda x: x["input"]) | rag_retriever,
    # ),
    # # If chat history, then we pass inputs to LLM chain, then to retriever
    #     templates.CONTEXTUALIZE_PROMPT | rag_llm | StrOutputParser() | rag_retriever)

    # # return retrieve_documents

    # # Stuff documents chain
    # # "Stuffs" or inputs the documents with the final prompt for the LLM
    # question_answer_chain = (
    #     RunnablePassthrough.assign(context=utils.format_docs)
    #     | templates.QA_PROMPT
    #     | rag_llm
    #     | StrOutputParser()
    # )

    # # RAG chain
    # # Combines the history-aware retriever and the question-answer chain
    # retrieval_chain = (
    #     RunnablePassthrough.assign(
    #         context=retrieve_documents
    #     )#.assign(answer=question_answer_chain)
    # )

    # return retrieval_chain
    history_aware_retriever = create_history_aware_retriever(
    rag_llm, rag_retriever, templates.CONTEXTUALIZE_PROMPT
    )
    question_answer_chain = create_stuff_documents_chain(rag_llm, templates.QA_PROMPT)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain



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

def reranker_chain(rag_llm):

    output_parser = RegexParser(
        regex=r"(.*?)\nScore: (.*)",
        output_keys=["answer", "score"],
    )
    prompt_template = (
        "Provide an answer of the question {input}. Output both your answer and a score of how confident "
        "you are. Context: {context}"
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context"],
        output_parser=output_parser,
    )
    llm_chain = LLMChain(llm=rag_llm, prompt=prompt)
    chain = MapRerankDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        rank_key="score",
        answer_key="answer",
    )
    return chain


def semantic_search(query:str, rag_llm:Runnable, rag_retriever:VectorStoreRetriever):
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

    response = chain.invoke(query)

    # TODO: Fetch Titles and metadata from metadata sql
    return response


