import time
import streamlit as st
import uuid
import os
import sys
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm
from dotenv import load_dotenv

root_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_dir)

from src.scripts.db_manager import DBManager
import rag
from chains import stuff_chain, reduce_chain, reranker_chain, semantic_search
from utils import load_vectorstore, load_llm, load_embedding

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")


class PageState:
    def __init__(self, session_id, subpage_name):
        self.subpage_name = subpage_name
        self.session_id = session_id
        self.messages = []

    def get_messages(self):
        return self.messages

    def add_message(self, message):
        self.messages.append(message)

    def clear_messages(self):
        self.messages = []


@st.cache_resource
def get_rag_components(_chain="stuffing"):
    print(f"Building RAG components for session with chain {_chain}")
    rag_llm = load_llm(temp=0.3)
    rag_retriever = load_vectorstore(QDRANT_URL, QDRANT_API_KEY).as_retriever()
    memory = rag.Memory()

    chain = None
    if _chain == "stuffing":
        chain = stuff_chain(rag_llm=rag_llm, rag_retriever=rag_retriever)
    elif _chain == "reduction":
        reduce_llm = load_llm(temp=0.05)
        chain = reduce_chain(qa_llm=rag_llm, reduce_llm=reduce_llm, rag_retriever=rag_retriever)
    elif _chain == "reranking":
        chain = reranker_chain(rag_llm=rag_llm)
    elif _chain == "hyde":
        pass
    else:
        raise ValueError(f"Invalid chain type: {_chain}")

    runnable = rag.build_runnable(chain, memory)
    return chain, memory, runnable

    # retriever = load_vectorstore(QDRANT_URL, QDRANT_API_KEY).as_retriever()
    # chain = rag.build_chain()
    # memory = rag.Memory()
    # runnable = rag.build_runnable(chain, memory)
    # vectorstore = rag.initialize_retriever().vectorstore
    # return chain, vectorstore, memory, runnable


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
