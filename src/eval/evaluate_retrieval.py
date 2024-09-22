import os
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
from tqdm import tqdm

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
COLLECTION_NAME = "arxiv_papers_RecursiveCharacterTextSplitter"

DATA_PATH = "data/qa_pairs.csv"


def get_arxiv_ids(docs):
    return [doc.metadata["arxiv_id"] for doc in docs]


def main():
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cuda"})
    vectorstore = QdrantVectorStore.from_existing_collection(
        collection_name=COLLECTION_NAME,
        embedding=embedding_function,
        url=os.getenv("QDRANT_URL"),
    )
    data_df = pd.read_csv(DATA_PATH)
    results = []
    for _, row in tqdm(data_df.iterrows()):
        target_arxiv_id = str(row["arxiv_id"])
        query = str(row["question"])
        similar_papers = vectorstore.similarity_search(query, k=10)
        arxiv_ids = get_arxiv_ids(similar_papers)
        top_10 = arxiv_ids[:10]
        top_5 = arxiv_ids[:5]
        top_3 = arxiv_ids[:3]
        top_1 = arxiv_ids[:1]
        top_10_accuracy = target_arxiv_id in top_10
        top_5_accuracy = target_arxiv_id in top_5
        top_3_accuracy = target_arxiv_id in top_3
        top_1_accuracy = target_arxiv_id in top_1
        results.append(
            {
                "query": query,
                "target_arxiv_id": target_arxiv_id,
                "top_10": top_10,
                "top_5": top_5,
                "top_3": top_3,
                "top_1": top_1,
                "top_10_accuracy": top_10_accuracy,
                "top_5_accuracy": top_5_accuracy,
                "top_3_accuracy": top_3_accuracy,
                "top_1_accuracy": top_1_accuracy,
            }
        )

    results_df = pd.DataFrame(results)
    top_10_accuracy = results_df["top_10_accuracy"].mean()
    top_5_accuracy = results_df["top_5_accuracy"].mean()
    top_3_accuracy = results_df["top_3_accuracy"].mean()
    top_1_accuracy = results_df["top_1_accuracy"].mean()
    print(f"Top 10 accuracy: {top_10_accuracy*100:.2f}%")
    print(f"Top 5 accuracy: {top_5_accuracy*100:.2f}%")
    print(f"Top 3 accuracy: {top_3_accuracy*100:.2f}%")
    print(f"Top 1 accuracy: {top_1_accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
