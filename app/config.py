# ------------------------------
# Indexing
# ------------------------------
VSDIR = r"qdrant/"
COLLECTION_NAME = "arxiv_papers_RecursiveCharacterTextSplitter"
# COLLECTION_NAME = "ilyi-test"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# ------------------------------
# OpenAI + LM Studio
# ------------------------------
LLM_TYPE = "ollama"
# LLM_TYPE = "lm-studio"

# ------------------------------
# Generation
# ------------------------------
QUESTION_TEMPLATE = "Condense the following question: {question}"
ANSWER_TEMPLATE = """Answer the question in the question's language based on your knowledge and the following context:
{context}

Question: {question}"""
