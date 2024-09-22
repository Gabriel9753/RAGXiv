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