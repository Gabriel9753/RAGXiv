import streamlit as st
import sys
import os
from collections import defaultdict

# --- Streamlit Setup ---
if "page_states" not in st.session_state:
    print("Setting session ID")
    st.session_state.page_states = {}

# --- PAGE SETUP ---
rag_chat_page = st.Page(
    "views/rag_chat.py",
    title="RAG Chat",
    icon=":material/chat:",
    default=True,
)

semantic_search_page = st.Page(
    "views/semantic_search.py",
    title="Semantic Search",
    icon=":material/search:",
)

paper_qa_page = st.Page(
    "views/paper_qa.py",
    title="Paper QA",
    icon=":material/question_mark:",
)

reference_graph_page = st.Page(
    "views/reference_graph.py",
    title="Reference Graph",
    icon="🧠",
)

settings_page = st.Page(
    "views/settings.py",
    title="Settings",
    icon=":material/settings:",
)

st.sidebar.title("RAGXiv")

# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
pg = st.navigation(pages=[rag_chat_page, semantic_search_page, paper_qa_page, reference_graph_page, settings_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
# pg = st.navigation(
#     {
#         "RAG": [rag_chat],
#         "Projects": [rag_chat],
#     }
# )


# --- SHARED ON ALL PAGES ---

# --- RUN NAVIGATION ---
pg.run()


st.sidebar.markdown(":grey-background[Made with 💚 by [Ilyi](https://github.com/ilyii) and [Gabriel](https://github.com/Gabriel9753)]")
