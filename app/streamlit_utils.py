import streamlit as st
import os
import sys
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm
import config
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

root_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_dir)

from src.scripts.db_manager import DBManager
import utils
import memory as mem
from chains import (
    stuff_chain,
    reduce_chain,
    reranker_chain,
    semantic_search_chain,
    hyde_chain,
    summarization_chain,
    paper_qa_chain
)
from utils import load_vectorstore, load_llm, load_embedding

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")


class PageState:
    def __init__(self, session_id, subpage_name):
        self.subpage_name = subpage_name
        self.session_id = session_id
        self.messages = []
        self.memory = None
        self.runnable = None
        self.chain = None
        self.cur_model = None

    def get_messages(self):
        return self.messages

    def add_message(self, message):
        self.messages.append(message)

    def clear_messages(self):
        self.messages = []

    def set_rag_components(self, chain, memory, runnable):
        self.chain = chain
        self.memory = memory
        self.runnable = runnable

    def get_rag_components(self):
        return self.chain, self.memory, self.runnable

    def set_model(self, model):
        self.cur_model = model

    def get_model(self):
        return self.cur_model

@st.cache_resource
def get_retreiver():
    retriever = load_vectorstore(QDRANT_URL, QDRANT_API_KEY).as_retriever()
    return retriever

def get_rag_components(_chain, _model):
    rag_llm = load_llm(temp=0.3, _model=_model)
    rag_retriever = get_retreiver()
    memory = mem.Memory()

    chain = None
    if _chain == "stuffing":
        chain = stuff_chain(rag_llm=rag_llm, rag_retriever=rag_retriever)
    elif _chain == "reduction":
        reduce_llm = load_llm(temp=0.05, _model=_model)
        chain = reduce_chain(qa_llm=rag_llm, reduce_llm=reduce_llm, rag_retriever=rag_retriever)
    elif _chain == "reranking":
        chain = reranker_chain(rag_llm=rag_llm, rag_retriever=rag_retriever)
    elif _chain == "hyde":
        chain = hyde_chain(rag_llm=rag_llm, rag_retriever=rag_retriever)
    elif _chain == "semantic_search":
        chain = semantic_search_chain(rag_llm=rag_llm, rag_retriever=rag_retriever)
    elif _chain == "summarization":
        chain = summarization_chain(rag_llm=rag_llm)
    elif _chain == "paper_qa":
        chain = paper_qa_chain(rag_llm=rag_llm)
    else:
        raise ValueError(f"Invalid chain type: {_chain}")
    runnable = utils.build_runnable(chain, memory)
    return chain, memory, runnable


@st.cache_resource
def get_db_manager():
    return DBManager()


def display_previous_messages(session_id):
    for message in st.session_state.page_states[session_id].get_messages():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and st.session_state.get("show_sources", True):
                with st.expander("Sources", expanded=False):
                    for source in message["sources"]:
                        st.markdown(source)


def get_retreived_papers(response):
    db_manager = get_db_manager()
    used_papers = defaultdict(dict)
    for doc in response["context"]:
        arxiv_id = doc.metadata.get("arxiv_id")
        page = doc.metadata.get("page")
        paper_metadata = db_manager.get_metadata_from_arxivid(arxiv_id)
        title = paper_metadata.get("title", None)
        used_papers[arxiv_id]["title"] = title
        if "pages" not in used_papers[arxiv_id]:
            used_papers[arxiv_id]["pages"] = set()
        used_papers[arxiv_id]["pages"].add(page)
    return used_papers


def build_used_papers_markdown(used_papers):
    papers = list(used_papers.items())[:5]  # Limit to 5 papers
    sources_list = []

    for arxiv_id, paper in papers:
        title = paper.get("title", arxiv_id)
        pages = paper.get("pages")
        pages_str = ", ".join([f"p. {page}" for page in sorted(pages)])
        source = f"[{title}](https://arxiv.org/abs/{arxiv_id}) (:grey[{pages_str}])"
        sources_list.append(source)

    return sources_list


def get_references(arxiv_id):
    db_manager = get_db_manager()
    references = db_manager.get_references_from_arxivid(arxiv_id)
    return references


def get_paper_metadata(arxiv_id):
    db_manager = get_db_manager()
    metadata = db_manager.get_metadata_from_arxivid(arxiv_id)
    return metadata


def get_authors(arxiv_id):
    db_manager = get_db_manager()
    authors = db_manager.get_authors_from_arxivid(arxiv_id)
    return authors


def scale_similarities(similarities):
    """Scale similarities to range [0, 1]"""
    min_sim = min(similarities.values())
    max_sim = max(similarities.values())
    if min_sim == max_sim:
        return {k: 1 for k in similarities}
    return {k: (v - min_sim) / (max_sim - min_sim) for k, v in similarities.items()}


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def get_title_similarity_values(main_title, titles, do_scale=True):
    embedder = load_embedding()
    title_embeddings = embedder.embed_documents(titles)
    main_title_embedding = embedder.embed_query(main_title)
    # Calculate cosine similarity between main_title and all titles without embedder
    similarity_values = {
        title: cosine_similarity(main_title_embedding, title_embedding)
        for title, title_embedding in zip(titles, title_embeddings)
    }

    similarity_values = scale_similarities(similarity_values) if do_scale else similarity_values
    return similarity_values


@st.cache_resource
def get_qdrant_client():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collection_name = config.COLLECTION_NAME
    return client, collection_name


def get_paper_content(arxiv_id):
    client, collection_name = get_qdrant_client()
    results = client.scroll(
        collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="metadata.arxiv_id", match=models.MatchValue(value=arxiv_id))],
        ),
        limit=99,
    )[0]
    content = " ".join([doc.payload["page_content"] for doc in results])
    return content


def get_similar_papers(prompt):
    vectorstore = load_vectorstore(QDRANT_URL, QDRANT_API_KEY)
    # retriever = vectorstore.as_retriever()
    response = vectorstore.similarity_search_with_score(prompt, k=5)
    return response


def get_predefined_prompt(prompt):
    # some predefined prompts
    if prompt.lower() == "/p1":
        prompt = "What is the method 'Stuffing' used for in a RAG-based chatbot?"
    elif prompt.lower() == "/p2":
        prompt = "What is the method 'Reduction' used for in a RAG-based chatbot?"
    elif prompt.lower() == "/p3":
        prompt = "What is the method 'Reranking' used for in a RAG-based chatbot?"
    elif prompt.lower() == "/p4":
        prompt = "What is the method 'HyDE' used for in a RAG-based chatbot?"
    elif prompt.lower() == "/p5":
        prompt = "In the context of learning with distribution shift, how does the complexity of the function class G, which represents the nuisance function, affect the performance of the predictor when the shift in the marginal distribution of y is significantly smaller than the shift in the joint distribution of (x, y)?"
    elif prompt.lower() == "/p6":
        prompt = "In the context of federated learning, what are the key differences between the proposed method and differentially private federated learning (DPFL) in terms of their mechanisms, certification goals, and technical contributions?"
    # some predefined arxiv ids
    elif prompt.lower() == "/a1":
        prompt = "2312.03511"
    elif prompt.lower() == "/a2":
        prompt = "2305.01644"
    elif prompt.lower() == "/a3":
        prompt = "2404.16689"
    elif prompt.lower() == "/a4":
        # Attention is All You Need
        prompt = "1706.03762"
    elif prompt.lower() == "/a5":
        # DINO
        prompt = "2104.14294"
    # some predefined questions to specific paper
    elif prompt.lower() == "/q1":
        prompt = "1706.03762 @ How does the use of multiple layers in both the encoder and decoder contribute to the model’s ability to capture complex dependencies in long sequences?"
    elif prompt.lower() == "/q2":
        prompt = "1706.03762 @ What is the effect of the number of attention heads in the model on the performance of the model?"
    elif prompt.lower() == "/q3":
        prompt = "1706.03762 @ Why are additive attention mechanisms less efficient than dot-product attention, particularly for larger dimensions?"
    else:
        prompt = prompt
    return prompt
