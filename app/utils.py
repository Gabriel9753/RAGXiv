import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain_core.runnables import RunnableWithMessageHistory
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(session_id="conversation_chain")

def format_docs(d:dict):
    """Formats the documents for prompt generation."""
    if isinstance(d, list):
        res = "\n\n".join([doc.page_content for doc in d])

    else:
        res = "\n\n".join(
            [doc.page_content for doc in d["context"]]
        )
    return res


def load_llm(temp:float=0.3):
    """Initialize and return the LLM based on the configured type."""
    if config.LLM_TYPE == "lm-studio":
        return ChatOpenAI(openai_api_base="http://localhost:5000/v1", openai_api_key="lm-studio", temperature=temp)
    elif config.LLM_TYPE == "ollama":
        return OllamaLLM(model="llama3.1:8b", temperature=temp)
    else:
        raise NotImplementedError(f"LLM type {config.LLM_TYPE} is not supported yet.")


def load_embedding():
    """Returns the embedding function based on the configuration."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME,
                                 model_kwargs={"device": "cpu"})


def load_vectorstore(qdrant_url:str, qdrant_api_key:str, hybrid:bool=False):
    """
    Initialize the retriever using the HuggingFace embeddings and Qdrant vectorstore.
    Returns:
        Retriever: A retriever for retrieving relevant documents.
    """
    embedding_function = load_embedding()

    if hybrid:

        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        vs = QdrantVectorStore.from_existing_collection(
            collection_name=config.COLLECTION_NAME,
            embedding=embedding_function,
            sparse_embedding=sparse_embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        """
        langchain_qdrant.qdrant.QdrantVectorStoreError: Existing Qdrant collection arxiv_papers_RecursiveCharacterTextSplitter does not contain sparse vectors named None. If you want to recreate the collection, set `force_recreate` parameter to `True`.
        """

    else:

        vs = QdrantVectorStore.from_existing_collection(
            collection_name=config.COLLECTION_NAME,
            embedding=embedding_function,
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

    return vs


def build_runnable(rag_chain, memory, keys: dict = None):
    """Build a runnable with message history"""

    if keys is None:
        keys = {"input_messages_key": "input", "history_messages_key": "chat_history", "output_messages_key": "answer"}

    return RunnableWithMessageHistory(
        rag_chain,
        memory.get_session_history,
        input_messages_key=keys["input_messages_key"],
        history_messages_key=keys["history_messages_key"],
        output_messages_key=keys["output_messages_key"],
    )

# @observe(name="chat()", as_type="generation")
def chat(rag, input_message, session_id, stream=False):
    """Chat with the model using the RAG chain."""
    # langfuse_handler = langfuse_context.get_current_langchain_handler()

    response = rag.invoke(
        {"input": input_message},
        config={"configurable": {"session_id": session_id}, "callbacks": [langfuse_handler]},
    )

    # langfuse_context.update_current_observation(
    #     input=input_message, output=response["answer"], session_id=session_id
    # )
    return response
