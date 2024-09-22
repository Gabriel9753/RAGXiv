import streamlit as st
from st_helpers.streamlit_utils import get_retreiver

# --- Streamlit Setup ---
st.set_page_config(page_title="RAGXiv", page_icon="🤖", layout="centered")

if "page_states" not in st.session_state:
    print("Setting session ID")
    st.session_state.page_states = {}
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True
if "llm" not in st.session_state:
    st.session_state.llm = "ollama/llama3.1:8b"
if "rag_method" not in st.session_state:
    st.session_state.rag_method = "stuffing"
get_retreiver()

# --- PAGE SETUP ---
rag_chat_page = st.Page(
    "views/rag_chat.py",
    title="RAG Chat",
    icon="💬",
    default=True,
)

semantic_search_page = st.Page(
    "views/semantic_search.py",
    title="Semantic Search",
    icon="🔍",
)

paper_qa_page = st.Page(
    "views/paper_qa.py",
    title="Paper QA",
    icon="❓",
)

summarization_page = st.Page(
    "views/summarization.py",
    title="Paper Summarization",
    icon="📝",
)

reference_graph_page = st.Page(
    "views/reference_graph.py",
    title="Reference Graph",
    icon="🧠",
)

settings_page = st.Page(
    "views/settings.py",
    title="Settings",
    icon="⚙️",
)

pg = st.navigation(
    pages=[rag_chat_page, semantic_search_page, summarization_page, paper_qa_page, reference_graph_page, settings_page]
)
st.sidebar.title("RAGXiv")
pg.run()
