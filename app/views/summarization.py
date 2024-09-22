import time
import streamlit as st
from streamlit_utils import (
    get_rag_components,
    PageState,
    get_retreived_papers,
    build_used_papers_markdown,
    display_previous_messages,
    get_references,
    get_paper_metadata,
    get_title_similarity_values,
    get_predefined_prompt,
    get_paper_content
)
import rag
import os
import hashlib
import re
from streamlit_agraph import agraph, Node, Edge, Config

# create an unique session id for this subpage
cur_file = os.path.basename(__file__)
page_name = os.path.basename(__file__).split(".")[0]
session_id = hashlib.md5(page_name.encode()).hexdigest()
# Initialize session state
if session_id not in st.session_state.page_states:
    st.session_state.page_states[session_id] = PageState(session_id, page_name)

chain, memory, runnable = get_rag_components(_chain="summarization")

# If the user clicks the "Clear chat history" button, clear the chat history
if st.sidebar.button("Clear chat history"):
    st.session_state.page_states[session_id].clear_messages()

# Display the chat history for the current session
display_previous_messages(session_id)


def valid_arxiv_id(arxiv_id):
    """Check if the arxiv_id is valid."""
    return bool(re.match(r"\d{4}\.\d{4,5}(v\d+)?", arxiv_id))


def valid_arxiv_url(arxiv_url):
    """Check if the arxiv_url is valid."""
    abs_url = "https://arxiv.org/abs/"
    pdf_url = "https://arxiv.org/pdf/"
    url_check = arxiv_url.startswith(abs_url) or arxiv_url.startswith(pdf_url)
    return url_check and valid_arxiv_id(arxiv_url.split("/")[-1])


def get_arxiv_id_from_url(arxiv_url):
    """Extract the arxiv_id from the arxiv_url."""
    return arxiv_url.split("/")[-1]


def get_summarization(arxiv_id, message_placeholder):
    full_response = ""
    full_paper_content = get_paper_content(arxiv_id)
    summarization = rag.chat(chain, full_paper_content, session_id)
    print(summarization)
    for chunk in summarization["answer"].split():
        full_response += chunk + " "
        time.sleep(0.001)
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    return full_response, summarization


prompt = st.chat_input("arxiv_id or arxiv_url")
if prompt:
    prompt = get_predefined_prompt(prompt)
    arxiv_id = get_arxiv_id_from_url(prompt) if valid_arxiv_url(prompt) else prompt
    if not valid_arxiv_id(arxiv_id):
        st.error("Invalid arXiv ID or URL. Please enter a valid arXiv ID or URL.")
        st.stop()

    # Add the user's prompt to the message history and display it
    st.session_state.page_states[session_id].add_message({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display the graph
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response, summarization = get_summarization(arxiv_id, message_placeholder)
        used_papers = get_retreived_papers(summarization)
        sources_list = build_used_papers_markdown(used_papers)

        # If enabled, display the sources which are expandable
        if st.session_state.get('show_sources', True):
            with st.expander("Sources", expanded=False):
                for source in sources_list:
                    st.markdown(source)

        # Add the response to the message history
        message = {"role": "assistant", "content": full_response, "sources": sources_list}
        st.session_state.page_states[session_id].add_message(message)
