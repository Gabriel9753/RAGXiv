import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableWithMessageHistory

import config
import utils
import memory
import templates
from chains import stuff_chain, reduce_chain, reranker, semantic_search, reranker_chain

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

if __name__ == "__main__":
    INPUT = "What is ChatGPT?"
    # SEMANTIC SEARCH
    # Retrieve documents with titles + links | short answer
    # ----
    # retriever = utils.load_vectorstore(QDRANT_URL, QDRANT_API_KEY).as_retriever()
    # llm = utils.load_llm(temp=0.3)
    # response = semantic_search(INPUT, rag_llm=llm, rag_retriever=retriever)
    # print(response) #TODO: Format + fetch metadata from SQL database
    # ----


    # # RAG (Chatbot)
    # # Prompt Enginnering for general answering + context-specific
    # # Stuffing
    # #TODO: Fix the chain, currently some key assignments are wrong
    # ----
    # retriever = utils.load_vectorstore(QDRANT_URL, QDRANT_API_KEY).as_retriever()
    # llm = utils.load_llm(temp=0.3)
    # mem = memory.Memory()
    # keys = {"input_messages_key": "input",
    #     "history_messages_key": "chat_history",
    #     "output_messages_key": "answer"}

    # session_id = "SAMPLE_SESSION_ID"
    # chain = stuff_chain(retriever, llm)
    # runnable = RunnableWithMessageHistory(
    #         chain,
    #         mem.get_session_history,
    #         input_messages_key=keys["input_messages_key"],
    #         history_messages_key=keys["history_messages_key"],
    #         output_messages_key=keys["output_messages_key"],
    #     )

    # response = runnable.invoke({"input":INPUT}, config={"configurable": {"session_id": session_id}})
    # print(response)
    # ----


    # # Reduce
    ## ----
    # retriever = utils.load_vectorstore(QDRANT_URL, QDRANT_API_KEY).as_retriever()

    # mem = memory.Memory()
    # keys = {"input_messages_key": "input",
    #     "history_messages_key": "chat_history",
    #     "output_messages_key": "answer"}

    # session_id = "SAMPLE_SESSION_ID"
    # chain = reduce_chain(qa_llm=utils.load_llm(temp=0.3), reduce_llm=utils.load_llm(temp=0.05), rag_retriever=retriever)
    # runnable = RunnableWithMessageHistory(
    #         chain,
    #         mem.get_session_history,
    #         input_messages_key=keys["input_messages_key"],
    #         history_messages_key=keys["history_messages_key"],
    #         output_messages_key=keys["output_messages_key"],
    #     )

    # response = runnable.invoke({"input":INPUT}, config={"configurable": {"session_id": session_id}})
    # print(response)
    # ----


    # Rerank
    # ----
    retriever = utils.load_vectorstore(QDRANT_URL, QDRANT_API_KEY).as_retriever()
    reranker_retriever = reranker(retriever)

    compressed_docs = reranker_retriever.invoke(INPUT)
    print(compressed_docs)

    chain = reranker_chain(rag_llm=utils.load_llm(temp=0.3))
    response = chain.invoke({"input": INPUT, "input_documents": retriever.invoke(INPUT)})
    print(response)

    # Paper QA
    # prompt + chain - How to only refer to this paper

    # Summarization
    # Paper ID -> Summary
    # Opt: Upload PDF -> Summary


