# Import necessary libraries
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import yaml
from dotenv import load_dotenv
import config

load_dotenv()

url = "https://1ed4f85b-722b-4080-97a7-afe8eab7ae7a.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def retriever():
    """
    Initialize the retriever using the HuggingFace embeddings and Chroma for persistence.

    Returns:
        Retriever: A retriever for retrieving relevant documents.
    """
    embedding_function = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})

    # Initialize the vectorstore using Chroma for persistence
    vectorstore = QdrantVectorStore.from_existing_collection(
        # path=config.CHROMADIR,
        collection_name="arxiv_demo",
        embedding=embedding_function,
        url=url,
        api_key=QDRANT_API_KEY,
    )

    return vectorstore.as_retriever()


if __name__ == "__main__":
    # Initialize components
    # llm, history_retriever, chain = initialize().values()
    yaml_path = "app/questions.yaml"

    retr = retriever()
    with open(yaml_path, "r", encoding="utf-8") as f:
        questions = yaml.load(f, Loader=yaml.FullLoader)["questions"]

    for i, question in enumerate(questions, 1):
        print(f"##################\n### Question {i} ###\n##################")
        q = question["question"]
        a = question["answer"]
        pdf = question["paper"]

        # Chat with the model
        output = retr.invoke(q)
        print(
            f'~~~ Question ~~~\n{q}\n\n~~~ Output ~~~\n{output}\n\n~~~ "Correct" Answer ~~~\n{a}\n\n~~~ Paper ~~~\n{pdf}\n'
        )
