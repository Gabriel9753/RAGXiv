import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.callback import CallbackHandler

from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
COLLECTION_NAME = "arxiv_papers_RecursiveCharacterTextSplitter"

langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


def format_docs(d: dict) -> str:
    """Formats the documents for prompt generation.

    Args:
        d (dict): The document dictionary.

    Returns:
        str: The formatted documents.
    """
    if isinstance(d, list):
        res = "\n\n".join([doc.page_content for doc in d])

    else:
        res = "\n\n".join([doc.page_content for doc in d["context"]])
    return res


def load_llm(temp: float = 0.3, _model=None):
    """
    Initialize and return the LLM (Large Language Model) based on the configured type.

    Parameters:
    temp (float): The temperature setting for the LLM, which controls the randomness of the output. Default is 0.3.
    _model (str): The identifier for the model to be loaded. Supported models include:
        - "lm-studio": Uses ChatOpenAI with a local API endpoint.
        - "ollama/qwen2.5:7b": Uses OllamaLLM with the "qwen2.5:7b" model.
        - "ollama/llama3.1:8b": Uses OllamaLLM with the "llama3.1:8b" model.
        - "gemini-1.5-flash": Uses ChatGoogleGenerativeAI with the "gemini-1.5-flash" model, requires GEMINI_API_KEY.

    Returns:
    An instance of the specified LLM.

    Raises:
    NotImplementedError: If the specified model type is not supported.
    Exception: If there is an error loading the "gemini-1.5-flash" model.
    """

    if "lm-studio" in _model:
        return ChatOpenAI(openai_api_base="http://localhost:5000/v1", openai_api_key="lm-studio", temperature=temp)
    elif "ollama/qwen2.5:7b" == _model:
        return OllamaLLM(model="qwen2.5:7b", temperature=temp)
    elif "ollama/llama3.1:8b" == _model:
        return OllamaLLM(model="llama3.1:8b", temperature=temp)
    elif "gemini-1.5-flash" == _model and GEMINI_API_KEY:
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=temp,
                api_key=GEMINI_API_KEY,
                max_tokens=1000,
                max_retries=2,
            )
        except Exception as e:
            raise Exception(f"Error loading the model: {e}")
    else:
        raise NotImplementedError(f"LLM type {_model} is not supported yet.")


def load_embedding(device: str = "cpu") -> HuggingFaceEmbeddings:
    """Returns the embedding function based on the configuration.

    Args:
        device (str): The device to use for the embeddings. Default is "cpu".

    Returns:
        HuggingFaceEmbeddings: The HuggingFace embeddings
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device})


def load_vectorstore(qdrant_url: str, qdrant_api_key: str, hybrid: bool = False) -> QdrantVectorStore:
    """
    Load a Qdrant vector store from an existing collection.
    Args:
        qdrant_url (str): The URL of the Qdrant instance.
        qdrant_api_key (str): The API key for accessing the Qdrant instance.
        hybrid (bool, optional): Whether to use hybrid retrieval mode with sparse embeddings. Defaults to False.
    Returns:
        QdrantVectorStore: An instance of QdrantVectorStore loaded from the existing collection.
    Raises:
        QdrantVectorStoreError: If the existing Qdrant collection does not contain the required sparse vectors
                                when hybrid mode is enabled.
    """
    embedding_function = load_embedding()

    if hybrid:
        raise NotImplementedError("Hybrid mode is not supported yet.")

        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        vs = QdrantVectorStore.from_existing_collection(
            collection_name=COLLECTION_NAME,
            embedding=embedding_function,
            sparse_embedding=sparse_embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key,
            retrieval_mode=RetrievalMode.HYBRID,
        )

    else:

        vs = QdrantVectorStore.from_existing_collection(
            collection_name=COLLECTION_NAME,
            embedding=embedding_function,
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

    return vs


def build_runnable(rag_chain: object, memory: object, keys: dict = None) -> RunnableWithMessageHistory:
    """Build a runnable with message history

    Args:
        rag_chain (object): The RAG chain object.
        memory (object): The memory object that provides session history.
        keys (dict, optional): A dictionary specifying the keys for input, history, and output messages. Defaults to None.

    Returns:
        RunnableWithMessageHistory: An instance of RunnableWithMessageHistory configured with the provided parameters.
    """

    if keys is None:
        keys = {"input_messages_key": "input", "history_messages_key": "chat_history", "output_messages_key": "answer"}

    return RunnableWithMessageHistory(
        rag_chain,
        memory.get_session_history,
        input_messages_key=keys["input_messages_key"],
        history_messages_key=keys["history_messages_key"],
        output_messages_key=keys["output_messages_key"],
    )

def chat(rag: object, input_message: str, session_id: str = None, trace_name: str = "chat()", context: dict = None) -> dict:
    """
    Chat with the model using the RAG chain.
    Args:
        rag (object): The RAG (Retrieval-Augmented Generation) model instance.
        input_message (str): The input message to send to the model.
        session_id (str, optional): The session ID for the chat. Defaults to None.
        trace_name (str, optional): The trace name for logging purposes. Defaults to "chat()".
        context (dict, optional): Additional context to provide to the model. Defaults to None.
    Returns:
        dict: The response from the model.
    """
    langfuse_handler.session_id = session_id
    langfuse_handler.trace_name = trace_name
    langfuse_handler.user_id = "user"

    if session_id is None:
        config = {"callbacks": [langfuse_handler]}
    else:
        config = {"configurable": {"session_id": session_id}, "callbacks": [langfuse_handler]}

    if context:
        response = rag.invoke({"input": input_message, "context": context}, config=config)
    else:
        response = rag.invoke({"input": input_message}, config=config)

    return response
