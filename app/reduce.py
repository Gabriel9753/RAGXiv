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







