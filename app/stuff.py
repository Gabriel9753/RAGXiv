import os
import uuid
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
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
    [("system", CONTEXTUALIZE_TEMPLATE), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [("system", QA_TEMPLATE), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)




def build_chain(rag_llm: ChatOpenAI, rag_retriever:VectorStoreRetriever):
    """Create the RAG chain using the retriever, prompts, and LLM."""




# @observe(name="build_runnable()", as_type="generation")

def build(rag_llm:Runnable, rag_retriever:VectorStoreRetriever, rag_memory:Memory, keys:dict = None):
    """Build a runnable with message history"""
    if keys is None:
        keys = {"input_messages_key": "input",
                "history_messages_key": "chat_history",
                "output_messages_key": "answer"}

    # History-aware retriever
    # Contextualizes the user input based on the chat history for the retrieval
    retrieve_documents: Runnable[Any, list[Document]] = RunnableBranch(
    (
        # Both empty string and empty list evaluate to False
        lambda x: not x.get("chat_history", False),
        # If no chat history, then we just pass input to retriever
        (lambda x: x["input"]) | rag_retriever,
    ),
    # If chat history, then we pass inputs to LLM chain, then to retriever
        CONTEXTUALIZE_PROMPT | rag_llm | StrOutputParser() | rag_retriever)


    # Stuff documents chain
    # "Stuffs" or inputs the documents with the final prompt for the LLM
    question_answer_chain = (
        RunnablePassthrough.assign(**{"context": utils.format_docs})
        | QA_PROMPT
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


if __name__ == "__main__":
    llm = utils.load_llm()
    vs = utils.load_vectorstore(QDRANT_URL, QDRANT_API_KEY)
    retriever = vs.as_retriever()
    memory = Memory()

    runnable = build(llm, retriever, memory)

    yaml_path = "app/questions.yaml"

    with open(yaml_path, "r", encoding="utf-8") as f:
        questions = yaml.load(f, Loader=yaml.FullLoader)["questions"]

    session_id = str(uuid.uuid4())
    for i, question in enumerate(questions, 1):
        print(f"##################\n### Question {i} ###\n##################")

        q = question["question"]
        a = question["answer"]
        pdf = question["paper"]

        # Chat with the model
        response = runnable.invoke(
            {"input": q},
            config={"configurable": {"session_id": session_id}},
        )


        # print(
        #     f'~~~ Question ~~~\n{q}\n\n~~~ Output ~~~\n{output["answer"]}\n\n~~~ "Correct" Answer ~~~\n{a}\n\n~~~ Paper ~~~\n{pdf}\n'
        # )
        print(response["answer"])