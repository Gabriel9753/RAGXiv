# ------------------------------
# Indexing
# ------------------------------
CHROMADIR = r"chroma/"
EMBEDDING_MODEL_NAME = "sentence-transformers/allenai-specter"

# ------------------------------
# OpenAI + LM Studio
# ------------------------------
LLM_TYPE = "ollama"  # "llm-studio" or "ollama"

# ------------------------------
# Generation
# ------------------------------
QUESTION_TEMPLATE = "Condense the following question: {question}"
ANSWER_TEMPLATE = """Answer the question in the question's language based on your knowledge and the following context:
{context}

Question: {question}"""
