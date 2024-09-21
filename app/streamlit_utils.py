import time
import streamlit as st
import uuid
import os
import sys
from collections import defaultdict

root_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_dir)

from src.scripts.db_manager import DBManager
import rag

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
def get_rag_components(id):
    chain = rag.build_chain()
    memory = rag.Memory()
    runnable = rag.build_runnable(chain, memory)
    vectorstore = rag.initialize_retriever().vectorstore
    return chain, vectorstore, memory, runnable

@st.cache_resource
def get_db_manager():
    return DBManager()

def display_previous_messages(session_id):
    for message in st.session_state.page_states[session_id].get_messages():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and st.session_state.get('show_sources', True):
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
        source = f"[{title}](https://arxiv.org/abs/{arxiv_id}) ({pages_str})"
        sources_list.append(source)

    return sources_list