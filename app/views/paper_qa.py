import time
import streamlit as st
from streamlit_utils import get_rag_components, PageState, get_retreived_papers, build_used_papers_markdown, display_previous_messages, get_predefined_prompt
from utils import chat
import os
import hashlib

import re
# create an unique session id for this subpage
cur_file = os.path.basename(__file__)
page_name = os.path.basename(__file__).split(".")[0]
session_id = hashlib.md5(page_name.encode()).hexdigest()

# Initialize session state
if session_id not in st.session_state.page_states:
    st.session_state.page_states[session_id] = PageState(session_id, page_name)

# Get the RAG components (specific chain), depending on the method chosen
# TODO: Chain for qa
chain, memory, runnable = get_rag_components(_chain="paper_qa")

# If the user clicks the "Clear chat history" button, clear the chat history
if st.sidebar.button("Clear chat history"):
    st.session_state.page_states[session_id].clear_messages()
    memory.clear()

# Display the chat history for the current session
display_previous_messages(session_id)

def qa_paper(arxiv_ids, question, message_placeholder):
    full_response = ""
    response = chat(runnable, question, session_id)
    for chunk in response["answer"].split():
        full_response += chunk + " "
        time.sleep(0.001)
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    return full_response, response

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

prompt = st.chat_input("[arxiv_id, arxiv_url, ...]@ Your question here")
# If the user has entered a prompt, chat with the assistant
if prompt:
    prompt = get_predefined_prompt(prompt)
    # index of first comma
    start_index = prompt.find("@")
    if start_index == -1:
        question = prompt.strip()
        arxiv_ids = []
    else:
        # Split the prompt by commas and remove leading/trailing whitespace
        provided_ids = prompt[:start_index].strip()
        question = prompt[start_index+1:].strip()
        if "," not in provided_ids and not valid_arxiv_id(provided_ids) and not valid_arxiv_url(provided_ids):
            st.warning("Please provide the arxiv_id or arxiv_url")
            st.stop()
        if "," not in provided_ids:
            arxiv_ids = [provided_ids]
        else:
            arxiv_ids = [arxiv_id.strip() for arxiv_id in provided_ids.split(",")]
        arxiv_ids = [arxiv_id for arxiv_id in arxiv_ids if valid_arxiv_id(arxiv_id) or valid_arxiv_url(arxiv_id)]
        arxiv_ids = [get_arxiv_id_from_url(arxiv_id) if valid_arxiv_url(arxiv_id) else arxiv_id for arxiv_id in arxiv_ids]
    # Add the user's prompt to the message history and display it
    st.session_state.page_states[session_id].add_message({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # First just get response
        full_response, response = qa_paper(arxiv_ids, question, message_placeholder)
        sources_list = ["MISSING"]
        # # In response the used papers are stored, get them and build markdown
        # if st.session_state.rag_method == "stuffing":
        #     used_papers = get_retreived_papers(response)
        #     sources_list = build_used_papers_markdown(used_papers)
        # elif st.session_state.rag_method == "reduction":
        #     sources_list = [response.get("context", "")]
        # message_placeholder.markdown(full_response)

        # # If enabled, display the sources which are expandable
        # if st.session_state.get('show_sources', True):
        #     with st.expander("Sources", expanded=False):
        #         for source in sources_list:
        #             st.markdown(source)

        # Add the response and sources to the message history
        message = {"role": "assistant", "content": full_response, "sources": sources_list}
        st.session_state.page_states[session_id].add_message(message)
