import time
import streamlit as st
import uuid
from streamlit_utils import get_rag_components, PageState, get_retreived_papers, build_used_papers_markdown, display_previous_messages
import rag
import os
import sys
import hashlib

# create an unique session id for this subpage
cur_file = os.path.basename(__file__)
page_name = os.path.basename(__file__).split(".")[0]
session_id = hashlib.md5(page_name.encode()).hexdigest()
chain, vectorstore, memory, runnable = get_rag_components(session_id)

# Initialize session state
if session_id not in st.session_state.page_states:
    st.session_state.page_states[session_id] = PageState(session_id, page_name)

if st.sidebar.button("Clear chat history"):
    st.session_state.page_states[session_id].clear_messages()
    memory.clear()

display_previous_messages(session_id)


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
    st.session_state.page_states[session_id].add_message({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response, response = normal_chat(prompt, message_placeholder)
        used_papers = get_retreived_papers(response)
        sources_list = build_used_papers_markdown(used_papers)

        message_placeholder.markdown(full_response)

        # Create a dropdown for sources if enabled
        if st.session_state.get('show_sources', True):
            with st.expander("Sources", expanded=False):
                for source in sources_list:
                    st.markdown(source)

        # Add the response and sources to the message history
        message = {"role": "assistant", "content": full_response, "sources": sources_list}
        st.session_state.page_states[session_id].add_message(message)
