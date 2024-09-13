import os

import chainlit as cl
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain_chroma import Chroma
from langchain_core.runnables import (RunnableLambda, RunnableParallel,
                                      RunnablePassthrough)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from config import Config

from operator import itemgetter
from typing import List, Tuple

from langchain.memory import ConversationBufferMemory

QUESTION_TEMPLATE = """
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

ANSWER_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""

# Create a memory instance


cfg = Config()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(temperature=.3, streaming=True, openai_api_base="http://localhost:5000/v1", openai_api_key="lm-studio")

    docstore = None

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma(persist_directory=cfg.chromapath, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    question_prompt = PromptTemplate.from_template(QUESTION_TEMPLATE)
    answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

    memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question", memory_key="chat_history"
    )

    # CHAIN
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | question_prompt
        | model
        | StrOutputParser(),
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: [x["docs"]],
        "question": itemgetter("question"),
    }

    answer = {
        "answer": final_inputs | answer_prompt | model,
        "docs": itemgetter("docs"),
    }

    # Create the final chain by combining the steps
    chain = loaded_memory | standalone_question | retrieved_documents | answer



    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")

    #TODO: Streaming
    # msg = cl.Message(content="")
    # await msg.send()

    res = chain.invoke(input={"question": message.content})
    print(res)
    answer = res["answer"].content
    docs = res["docs"]
    text_elements = [cl.Text(content=doc.page_content, name=f'{doc.metadata["source"]} - {doc.metadata["page"]}', display="side") for doc in docs]
    sourcenames = [el.name for el in text_elements]
    if sourcenames:
        answer += f"\nSources: {', '.join(sourcenames)}"
    else:
        answer += "\nNo sources found"
    await cl.Message(content=answer, elements=text_elements).send()




