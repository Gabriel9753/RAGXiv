from datasets import load_dataset
from langchain_openai.chat_models import ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas.run_config import RunConfig


# Load dataset
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2", trust_remote_code=True)["eval"]

# Initialize LLM and embeddings
llm = ChatOpenAI(openai_api_base="http://localhost:5000/v1", openai_api_key="lm-studio")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/allenai-specter", model_kwargs={"device": "cpu"})

# Metrics you chose
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

# Util function to init Ragas Metrics
def init_ragas_metrics(metrics, llm, embedding):
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = embedding
        run_config = RunConfig()
        metric.init(run_config)

# Initialize metrics
init_ragas_metrics(
    metrics,
    llm=LangchainLLMWrapper(llm),
    embedding=LangchainEmbeddingsWrapper(emb),
)

# Evaluate
result = evaluate(
    amnesty_qa,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)

# Print results
print(result)
df = result.to_pandas()
print(df.head())
