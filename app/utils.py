import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import FastEmbedSparse, RetrievalMode


def format_docs(d:dict):
    """Formats the documents for prompt generation."""
    if isinstance(d, list):
        res = "\n\n".join([doc.page_content for doc in d])

    else:
        print(d)
        print(d.keys())
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


