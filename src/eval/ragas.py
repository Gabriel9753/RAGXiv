#!/usr/bin/env python
# coding: utf-8

# Required imports and loading of dataset
from datasets import load_from_disk
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableWithMessageHistory
import utils
import memory
import chains
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import StuffDocumentsChain, LLMChain
from datasets import Dataset
import ragas
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

# Load environment variables
load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# Load the dataset
ds = load_from_disk("./qa_dataset")

# Load vector store and retriever
vs = utils.load_vectorstore(QDRANT_URL, QDRANT_API_KEY)
retriever = vs.as_retriever()

# Load language model and memory
llm = utils.load_llm(temp=0.3)
memory = memory.Memory()

# Define prompt template for question-answering task
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Define a chain with recursive character splitting and LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt)
document_prompt = PromptTemplate(input_variables=["page_content"], template="{page_content}")
chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name="context"
)

# Prepare the questions and ground truths
questions = ds["questions"]
ground_truth = ds["ground_truth"]
if isinstance(ground_truth[0], list):
    ground_truth = [gt[0] for gt in ground_truth]

# Initialize an empty data dictionary for the new dataset
data = {"question": [], "answer": [], "contexts": [], "ground_truth": ground_truth}

# Simulate RAG process: querying and retrieving documents
for query in questions:
    docs = retriever.invoke(query, top_k=4)  # Retrieve top-k documents
    answer = chain.invoke({"question": query, "input_documents": docs})  # Get LLM answer
    context = [doc.page_content for doc in docs]  # Extract context

    # Append results
    data["question"].append(query)
    data["answer"].append(answer["output_text"])
    data["contexts"].append(context)

# Create a dataset from the dictionary
dataset = Dataset.from_dict(data)

# Convert dataset to pandas DataFrame for further processing
dataset = dataset.to_pandas()
dataset["ground_truth"] = [gt[0] for gt in dataset["ground_truth"]]

# Evaluate the dataset using RAGAS metrics (Context Precision, Recall, Answer Relevancy, Faithfulness)
result = ragas.evaluate(
    dataset=Dataset.from_pandas(dataset),
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy
    ],
    llm=llm,
)

# Convert the results to a pandas DataFrame
df = result.to_pandas()

# Visualize results using a heatmap
heatmap_data = df[['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']]
cmap = LinearSegmentedColormap.from_list('green_red', ['red', 'green'])

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=.5, cmap=cmap)
plt.yticks(ticks=range(len(df['question'])), labels=df['question'], rotation=0)
plt.show()

# Calculate the mean of each metric
means = df[["faithfulness", "context_precision", "context_recall", "answer_relevancy"]].mean()
print(means)

# Save the evaluation results to CSV
df.to_csv("evaluation_results.csv", index=False)
