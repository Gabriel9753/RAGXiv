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
