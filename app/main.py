import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableWithMessageHistory

import config
import utils
import memory
import templates
import chains

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

if __name__ == "__main__":
    INPUT = "What is ChatGPT?"
    # SEMANTIC SEARCH
    # Retrieve documents with titles + links | short answer
    # ----
    # retriever = utils.load_vectorstore(QDRANT_URL, QDRANT_API_KEY, hybrid=True).as_retriever()
    # llm = utils.load_llm(temp=0.3)
    # response = chains.semantic_search(INPUT, rag_llm=llm, rag_retriever=retriever)
    # print(response) #TODO: Format + fetch metadata from SQL database
    # ----


    # # RAG (Chatbot)
    # # Prompt Enginnering for general answering + context-specific
    # # Stuffing
    # #TODO: Fix the chain, currently some key assignments are wrong
    # ----
    # retriever = utils.load_vectorstore(QDRANT_URL, QDRANT_API_KEY).as_retriever()

    # chain = chains.stuff_chain(rag_llm=utils.load_llm(temp=0.3),
    #                        rag_retriever=retriever)

    # runnable = RunnableWithMessageHistory(
    #     chain,
    #     memory.Memory().get_session_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    #     output_messages_key="answer",
    # )
    # response = runnable.invoke({"input": INPUT}, #
    #                            config={"configurable": {"session_id": "SAMPLE_SESSION_ID"}})
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
    # chain = chains.reduce_chain(qa_llm=utils.load_llm(temp=0.3), reduce_llm=utils.load_llm(temp=0.05), rag_retriever=retriever)
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
    # retriever = utils.load_vectorstore(QDRANT_URL, QDRANT_API_KEY).as_retriever()
    # reranker_retriever = reranker(retriever)

    # chain = chains.reranker_chain(rag_llm=utils.load_llm(temp=0.3),
    #                        rag_retriever=retriever)

    # runnable = RunnableWithMessageHistory(
    #     chain,
    #     memory.Memory().get_session_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    #     output_messages_key="answer",
    # )
    # response = runnable.invoke({"input": INPUT}, config={"configurable": {"session_id": "SAMPLE_SESSION_ID"}})
    # print(response)

    # Paper QA
    # prompt + chain - How to only refer to this paper
    # sum_chain = chains.paper_qa_chain(rag_llm=utils.load_llm(temp=0.3))

    # runnable = RunnableWithMessageHistory(
    #     sum_chain,
    #     memory.Memory().get_session_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    #     output_messages_key="answer",
    # )
    # paper="PAPER_ID"
    # content = "FETCH PAPER CONTENT by RESOLVING `paper` PARAM"

    # response = runnable.invoke({"input": INPUT, "context":content}, config={"configurable": {"session_id": "SAMPLE_SESSION_ID"}})
    # print(response)
    # response = runnable.invoke({"input": INPUT, "context":content}, config={"configurable": {"session_id": "SAMPLE_SESSION_ID"}})
    # print(response)


    # # Summarization
    # # Paper ID -> Summary
    # sum_chain = chains.summarization_chain(rag_llm=utils.load_llm(temp=0.3))

    # runnable = RunnableWithMessageHistory(
    #     sum_chain,
    #     memory.Memory().get_session_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    #     # output_messages_key="answer",
    # )
    # paper="PAPER_ID"
    # content = "FETCH PAPER CONTENT by RESOLVING `paper` PARAM"

    # response = runnable.invoke({"input": content}, config={"configurable": {"session_id": "SAMPLE_SESSION_ID"}})
    # print(response)

    # # HyDE

    retriever = utils.load_vectorstore(QDRANT_URL, QDRANT_API_KEY).as_retriever()
    chain = chains.hyde_chain(rag_llm=utils.load_llm(temp=0.3), rag_retriever=retriever)

    runnable = RunnableWithMessageHistory(
        chain,
        memory.Memory().get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    response = runnable.invoke({"input": INPUT}, config={"configurable": {"session_id": "SAMPLE_SESSION_ID"}})
    print(response)



