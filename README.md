# RAGXiv
RAGXiv is a project that integrates data from arXiv with semantic metadata using various tools like Semantic Scholar and Qdrant, along with machine learning models for text analysis. This project handles the full pipeline from downloading arXiv data, processing it, to launching a web application for interactive exploration.

## Installation

### Prerequisites
- `python=3.11`
- `poetry`
- `docker` + `docker-compose`
- `Ollama` | `lm-studio` | other LLM


### Steps

Create a virtual environment:
```bash
python -m venv .venv
```

Activate the virtual environment:

- Windows
```bash
.venv\Scripts\activate 
```

- Linux
```bash
source .venv/bin/activate
```

Install poetry via pip:  
```bash 
pip install poetry
```

Install dependencies specified in `pyproject.toml`:
```bash
poetry install
```

#### Docker
Pull and compose docker
```bash
docker-compose up -d
```

### Environment Variables
Create a `.env` file in the root directory with the following variables:
```bash
DATABASE_URL=$DATABASE_URL
QDRANT_API_KEY=$QDRANT_API_KEY
QDRANT_URL=http://localhost:6333
GEMINI_API_KEY=$GEMINI_API_KEY
LANGFUSE_PK=$LANGFUSE_PK
LANGFUSE_SK=$LANGFUSE_SK
LANGFUSE_HOST="https://cloud.langfuse.com"
```

### Data loading and cleaning

1. Download the JSON snapshot from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) containing all papers.

2. Run `src/data_processing/kaggle_data_processing.ipynb` to clean the data and create a sample dataset.

3. Execute `src/data_processing/request_paper_metadata.py` to fetch metadata (authors, references, etc.) for the papers from [Semantic Scholar](https://www.semanticscholar.org/).

4. Run `src/data_processing/download_papers.py` to download the complete papers from arXiv.

5. Execute `src/data_processing/eda.py` to visualize the data and perform exploratory data analysis.

6. Run `src/scripts/index.py` to index the papers in Qdrant and store the metadata in the database.

7. Start the web application by running `streamlit run src/app.py`!

These steps outline the process from data acquisition to running the final web application, including data processing, metadata retrieval, paper downloading, data visualization, indexing, and launching the Streamlit app.
