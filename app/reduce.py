import os
import uuid
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import VectorStoreRetriever
# from langchain_core.documents import Document
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser

import utils
from memory import Memory


load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

#TODO: Move to templates.py
CONTEXTUALIZE_TEMPLATE = (
    "Given a chat history and the latest user question, \
    reformulate the question to be understood independently."
)
QA_TEMPLATE = (
    "You are an assistant for question-answering tasks. \
    Use the provided context to answer the question concisely.\n\n \
    Context: {context}"
)

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", CONTEXTUALIZE_TEMPLATE),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [("system", QA_TEMPLATE),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")]
)

COMBINE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", "Provide a ombine the following documents into a single document: {context}."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")]
)

COLLAPSE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", "Provide the context in a collapsed, precise form: \n\n{context}."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")]

)

def build(qa_llm:Runnable, reduce_llm:Runnable,  rag_retriever:VectorStoreRetriever, rag_memory:Memory, keys:dict = None):
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


    if keys is None:
        keys = {"input_messages_key": "input",
                "history_messages_key": "chat_history",
                "output_messages_key": "answer"}

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
        CONTEXTUALIZE_PROMPT | reduce_llm | StrOutputParser() | rag_retriever)

    # Collapse documents chain
    # If max_tokens is exceeded, collapse the document string through summarization with llm
    collapse_chain = (
        {"input": lambda x: x["input"], "chat_history": lambda x: x["chat_history"]}
        | RunnablePassthrough.assign(
            context=retrieve_documents | utils.format_docs # Combine documents
        )
        | COLLAPSE_PROMPT
        | reduce_llm
        | StrOutputParser()
    )

    # Question-Answer chain
    qa_chain = (
        {"context": collapse_chain,
         "input": lambda x: x["input"],
         "chat_history": lambda x: x["chat_history"]}
        | QA_PROMPT
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

    # return retrieval_chain
    # Runnable with message history
   # Incorporates the chat history into the runnable
    return RunnableWithMessageHistory(
        retrieval_chain,
        rag_memory.get_session_history,
        input_messages_key=keys["input_messages_key"],
        history_messages_key=keys["history_messages_key"],
        output_messages_key=keys["output_messages_key"],
    )

# def init_langfuse():
#     from langfuse import Langfuse
#     langfuse = Langfuse()
#     print("Langfuse available: ",langfuse.auth_check())
#     from langfuse.decorators import langfuse_context, observe
#     from langfuse.callback import CallbackHandler
#     langfuse_handler = CallbackHandler(session_id="conversation_chain")
#     return langfuse

# def chat(chain):



if __name__ == "__main__":
    qa_llm = utils.load_llm(temp=0.3)
    preprocess_llm = utils.load_llm(temp=0.05)
    vs = utils.load_vectorstore(QDRANT_URL, QDRANT_API_KEY)
    retriever = vs.as_retriever()
    memory = Memory()

    runnable = build(qa_llm,preprocess_llm, retriever, memory)

    yaml_path = "app/questions.yaml"

    with open(yaml_path, "r", encoding="utf-8") as f:
        questions = yaml.load(f, Loader=yaml.FullLoader)["questions"]

    session_id = str(uuid.uuid4())



    # ----------------- RERANKING TBD -----------------
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    compressed_docs = compression_retriever.invoke("What is ChatGPT?")
    print(compressed_docs)
    def pretty_print_docs(docs):
        print(
            f"\n{'-' * 100}\n".join(
                [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
            )
        )
    pretty_print_docs(compressed_docs)
    # for i, question in enumerate(questions, 1):
    #     print(f"##################\n### Question {i} ###\n##################")

    #     q = question["question"]
    #     a = question["answer"]
    #     pdf = question["paper"]

    #     # Chat with the model
    #     response = runnable.invoke(
    #         {"input": q},
    #         config={"configurable": {"session_id": session_id}},
    #     )

    #     # print(
    #     #     f'~~~ Question ~~~\n{q}\n\n~~~ Output ~~~\n{output["answer"]}\n\n~~~ "Correct" Answer ~~~\n{a}\n\n~~~ Paper ~~~\n{pdf}\n'
    #     # )
    #     print("INPUT: ", response["input"])
    #     print("CONTEXT: ", response["context"])
    #     print("OUTPUT: ", response["answer"])



# SEMANTIC SEARCH
# Retrieve documents with titles + links | short answer

# RAG (Chatbot)
# Prompt Enginnering for general answering + context-specific
    # Stuffing
    # Reduce
    # Rerank

# Paper QA
# prompt + chain - How to only refer to this paper

# Summarization
# Paper ID -> Summary
# Opt: Upload PDF -> Summary




