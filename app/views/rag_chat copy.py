import time
import streamlit as st
import uuid
from streamlit_utils import get_rag_components, get_db_manager, PageState
import rag
import os
import sys
import hashlib
from collections import defaultdict

db_manager = get_db_manager()

# create an unique session id for this subpage
cur_file = os.path.basename(__file__)
page_name = os.path.basename(__file__).split(".")[0]
session_id = hashlib.md5(page_name.encode()).hexdigest()
chain, vectorstore, memory, runnable = get_rag_components(session_id)

# Initialize session state
if session_id not in st.session_state.page_states:
    st.session_state.page_states[session_id] = PageState(session_id, page_name)

for message in st.session_state.page_states[session_id].get_messages():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def get_retreived_papers(response):
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
    # markdown should be a nice light grey box (dropdown for expandability)
    # with a list of papers and their links
    # if a paper has a title, display it, otherwise display the arxiv_id
    # if more than one page is used, list all pages, otherwise just list the page
    # max 5 papers

    papers = set()
    used_papers_markdown = ""
    for arxiv_id, paper in used_papers.items():
        papers.add(arxiv_id)
        title = paper.get("title", arxiv_id)
        pages = paper.get("pages")
        if len(pages) > 1:
            pages_str = ", ".join([f"p. {page}" for page in sorted(pages)])
        else:
            pages_str = f"p. {pages.pop()}"
        used_papers_markdown += f"[{title}](https://arxiv.org/abs/{arxiv_id}) ({pages_str})\n"

        if len(papers) >= 2:
            break
    return used_papers_markdown


def normal_chat(prompt, message_placeholder):
    full_response = ""
    response = rag.chat(runnable, prompt, session_id)
    for chunk in response["answer"].split():
        full_response += chunk + " "
        time.sleep(0.001)
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    return full_response, response




prompt = st.chat_input("Ask me anything about the papers in the knowledge base")
if prompt:
    # st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.page_states[session_id].add_message({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response, response = normal_chat(prompt, message_placeholder)
        used_papers = get_retreived_papers(response)
        used_papers_markdown = build_used_papers_markdown(used_papers)
        full_response += "\n\n*Sources:* " + used_papers_markdown
        message_placeholder.markdown(full_response)

    # st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.page_states[session_id].add_message({"role": "assistant", "content": full_response})
