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

# print("Langfuse available:", langfuse_handler.auth_check())


def format_docs(d: dict):
    """Formats the documents for prompt generation."""
    if isinstance(d, list):
        res = "\n\n".join([doc.page_content for doc in d])

    else:
        res = "\n\n".join([doc.page_content for doc in d["context"]])
    return res


def load_llm(temp: float = 0.3, _model=None):
    """Initialize and return the LLM based on the configured type."""
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


def load_embedding():
    """Returns the embedding function based on the configuration."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})


def load_vectorstore(qdrant_url: str, qdrant_api_key: str, hybrid: bool = False):
    """
    Initialize the retriever using the HuggingFace embeddings and Qdrant vectorstore.
    Returns:
        Retriever: A retriever for retrieving relevant documents.
    """
    embedding_function = load_embedding()

    if hybrid:

        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        vs = QdrantVectorStore.from_existing_collection(
            collection_name=COLLECTION_NAME,
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
            collection_name=COLLECTION_NAME,
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


def chat(rag, input_message, session_id=None, trace_name="chat()", context=None):
    """Chat with the model using the RAG chain."""
    print(rag)
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
