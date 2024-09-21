import streamlit as st
import sys
import os
from collections import defaultdict

# --- Streamlit Setup ---
if "page_states" not in st.session_state:
    print("Setting session ID")
    st.session_state.page_states = {}

# --- PAGE SETUP ---
rag_chat = st.Page(
    "views/rag_chat.py",
    title="RAG Chat",
    icon=":material/account_circle:",
    default=True,
)
# project_1_page = st.Page(
#     "views/rag_chat copy.py",
#     title="Sales Dashboard",
#     icon=":material/bar_chart:",
# )
settings_page = st.Page(
    "views/settings.py",
    title="Settings",
    icon=":material/settings:",
)


# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
pg = st.navigation(pages=[rag_chat, settings_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
# pg = st.navigation(
#     {
#         "RAG": [rag_chat],
#         "Projects": [rag_chat],
#     }
# )


# --- SHARED ON ALL PAGES ---
st.sidebar.markdown("Made with ❤️ by [Sven](https://youtube.com/@codingisfun)")

# Display the session ID in the sidebar
# st.sidebar.text(f"Session ID: {st.session_state.session_id}")

# --- RUN NAVIGATION ---
pg.run()
