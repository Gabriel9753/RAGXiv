import os
import sys
from typing import List, Tuple
import pandas as pd
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import IndexConfig
from data_processing.data_utils import load_data

load_dotenv()

MAX_PAPERS = 100
OUT_PATH = "data/qa_pairs.csv"

SYSTEM_TEMPLATE = """Formulate a question that requires a nuanced or detailed answer,
which can be derived from the context of the paper but could also be found by exploring related literature.
The question should not explicitly mention the paper, authors, or any generic terms like contributions or summaries.
Avoid asking for simple facts or definitions. The question should be subtle,
requiring contextual understanding without directly referencing the paper.

Do not introduce the question or answer with phrases like ‘this is a question’ or ‘this is an answer.’
Do not refer to ‘the paper’ or ‘the authors.’
The question should be 2-3 sentences long, and the answer should be 1-2 sentences long.
The format must be:
Question: ...
Answer: ..."""

def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()

def extract_content(pdf: List[Document], start_page: int = 3, end_page: int = 10) -> str:
    content = [p.page_content for p in pdf if start_page <= p.metadata["page"] <= end_page]
    return "\n".join(content)

def parse_response(response: str) -> Tuple[str, str]:
    parts = response.split("Question: ")
    if len(parts) != 2:
        raise ValueError("Invalid response format")
    question_answer = parts[1].split("Answer: ")
    if len(question_answer) != 2:
        raise ValueError("Invalid response format")
    return question_answer[0].strip(), question_answer[1].strip()

def create_qa_chain():
    # llm = OllamaLLM(model="qwen2.5:7b", temperature=0.1)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
        # other params...
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human", "{paper_content}")
    ])

    chain = (
        {"paper_content": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def process_paper(pdf_path: str, qa_chain) -> Tuple[str, str, str]:
    try:
        pdf = load_pdf(pdf_path)
        content = extract_content(pdf)

        response = qa_chain.invoke(content)
        question, answer = parse_response(response)

        return pdf_path, question, answer, content

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return pdf_path, "", "", ""

def main():
    papers_df, _, _ = load_data(drop_missing=True)
    qa_chain = create_qa_chain()

    results = []

    for i, row in tqdm(papers_df.iterrows(), total=min(MAX_PAPERS, len(papers_df)), desc="Processing papers"):
        if i >= MAX_PAPERS:
            break

        pdf_path, question, answer, content = process_paper(row["pdf_path"], qa_chain)
        arxiv_id = row["arxiv_id"]
        print(f"Processing paper {arxiv_id}")
        if question and answer:
            results.append((arxiv_id, pdf_path, question, answer, content))
            print(f"PDF: {pdf_path}")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print("-" * 50)

    qa_pairs_df = pd.DataFrame(results, columns=["arxiv_id", "pdf_path", "question", "answer", "content"])

    qa_pairs_df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"QA pairs saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
