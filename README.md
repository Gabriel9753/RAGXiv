# RAGXiv

python=3.11
python -m venv .venv
.venv\Scripts\activate | source .venv/bin/activate
pip install poetry
poetry install

### Docker

docker-compose up -d

### Guardrails

guardrails hub install hub://guardrails/toxic_language --quiet
guardrails hub install hub://guardrails/competitor_check --quiet

### Environment Variables

```bash
DATABASE_URL=$DATABASE_URL
QDRANT_API_KEY=$QDRANT_API_KEY
QDRANT_URL=http://localhost:6333
GEMINI_API_KEY=$GEMINI_API_KEY
LANGFUSE_PK=$LANGFUSE_PK
LANGFUSE_SK=$LANGFUSE_SK
LANGFUSE_HOST="https://cloud.langfuse.com"
```

Here's a reworked version of the steps in English:

### Steps

1. Download the JSON snapshot from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) containing all papers.

2. Run `src/data_processing/kaggle_data_processing.ipynb` to clean the data and create a sample dataset.

3. Execute `src/data_processing/request_paper_metadata.py` to fetch metadata (authors, references, etc.) for the papers from [Semantic Scholar](https://www.semanticscholar.org/).

4. Run `src/data_processing/download_papers.py` to download the complete papers from arXiv.

5. Execute `src/data_processing/eda.py` to visualize the data and perform exploratory data analysis.

6. Run `src/scripts/index.py` to index the papers in Qdrant and store the metadata in the database.

7. Start the web application by running `streamlit run src/app.py`!

These steps outline the process from data acquisition to running the final web application, including data processing, metadata retrieval, paper downloading, data visualization, indexing, and launching the Streamlit app.
