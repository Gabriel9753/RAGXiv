import time
import streamlit as st
from streamlit_utils import get_rag_components, PageState, get_retreived_papers, build_used_papers_markdown, display_previous_messages, get_predefined_prompt
import rag
import os
import hashlib

# create an unique session id for this subpage
cur_file = os.path.basename(__file__)
page_name = os.path.basename(__file__).split(".")[0]
session_id = hashlib.md5(page_name.encode()).hexdigest()

# Initialize session state
if session_id not in st.session_state.page_states:
    st.session_state.page_states[session_id] = PageState(session_id, page_name)

# Get the RAG components (specific chain), depending on the method chosen
# TODO: Add missing chains
chain, memory, runnable = get_rag_components()#_chain="similar")

# If the user clicks the "Clear chat history" button, clear the chat history
if st.sidebar.button("Clear chat history"):
    st.session_state.page_states[session_id].clear_messages()
    memory.clear()

# Display the chat history for the current session
display_previous_messages(session_id)

def get_similar(prompt, message_placeholder):
    full_response = ""
    response = rag.chat(runnable, prompt, session_id)
    for chunk in response["answer"].split():
        full_response += chunk + " "
        time.sleep(0.001)
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    return full_response, response

prompt = st.chat_input("Give me some context to find similar papers")
# If the user has entered a prompt, chat with the assistant
if prompt:
    prompt = get_predefined_prompt(prompt)
    # Add the user's prompt to the message history and display it
    st.session_state.page_states[session_id].add_message({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # First just get response
        full_response, response = get_similar(prompt, message_placeholder)
        # In response the used papers are stored, get them and build markdown
        used_papers = get_retreived_papers(response)
        sources_list = build_used_papers_markdown(used_papers)
        message_placeholder.markdown(full_response)

        # If enabled, display the sources which are expandable
        if st.session_state.get('show_sources', True):
            with st.expander("Sources", expanded=False):
                for source in sources_list:
                    st.markdown(source)

        # Add the response and sources to the message history
        message = {"role": "assistant", "content": full_response, "sources": sources_list}
        st.session_state.page_states[session_id].add_message(message)
