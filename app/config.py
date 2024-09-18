# ------------------------------
# Indexing
# ------------------------------
CHROMADIR = r"chroma/"
# CHROMADIR = r"D:\Personal\OneDrive\Gabrilyi\arxiv_project\final_embeddings\chroma_data\all-MiniLM-L12-v2"

EMBEDDING_MODEL_NAME = "sentence-transformers/allenai-specter"
TEXT_SPLITTER_ARGS = {
    "chunk_size": 300,
    "chunk_overlap": 100,
    "length_function": len,
    "add_start_index": True,
}

# ------------------------------
# OpenAI + LM Studio
# ------------------------------
OPENAI_API_BASE = "http://localhost:5000/v1"
OPENAI_API_KEY = "lm-studio"

# ------------------------------
# Generation
# ------------------------------
QUESTION_TEMPLATE = "Condense the following question: {question}"
ANSWER_TEMPLATE = """Answer the question in the question's language based on your knowledge and the following context:
{context}

Question: {question}"""
