import time
import streamlit as st
from st_helpers.streamlit_utils import (
    get_rag_components,
    PageState,
    display_previous_messages,
    get_predefined_prompt,
    get_paper_content,
    get_paper_metadata,
    valid_arxiv_id,
    valid_arxiv_url,
    get_arxiv_id_from_url,
    build_url,
)
from rag.utils import chat
import os
import hashlib

# create an unique session id for this subpage
cur_file = os.path.basename(__file__)
page_name = os.path.basename(__file__).split(".")[0]
session_id = hashlib.md5(page_name.encode()).hexdigest()

# Initialize session state
if session_id not in st.session_state.page_states:
    st.session_state.page_states[session_id] = PageState(session_id, page_name)
    chain, memory, runnable = get_rag_components(_chain="paper_qa", _model=st.session_state.llm)
    st.session_state.page_states[session_id].set_rag_components(chain, memory, runnable)
    st.session_state.page_states[session_id].set_model(st.session_state.llm)

# check if the model is set
if st.session_state.llm != st.session_state.page_states[session_id].get_model():
    st.session_state.page_states[session_id].set_model(st.session_state.llm)
    chain, memory, runnable = get_rag_components(_chain=st.session_state.rag_method, _model=st.session_state.llm)
    st.session_state.page_states[session_id].set_rag_components(chain, memory, runnable)

chain, memory, runnable = st.session_state.page_states[session_id].get_rag_components()

st.title(":rainbow[Paper QA]")
st.markdown("Welcome to the Paper QA! Enter an arXiv ID or URL and ask a question about the paper.")
st.markdown("Example: `https://arxiv.org/abs/1706.03762 @ What is the main contribution of this paper?`")
st.markdown("---")

# If the user clicks the "Clear chat history" button, clear the chat history
if st.sidebar.button("Clear chat history"):
    st.session_state.page_states[session_id].clear_messages()
    memory.clear()

st.sidebar.markdown(f"`Using model: {st.session_state.page_states[session_id].get_model()}`")

# Display the chat history for the current session
display_previous_messages(session_id)


def qa_paper(question, paper_content, message_placeholder):
    full_response = ""
    response = chat(runnable, question, session_id, trace_name="qa_paper", context=paper_content)
    for chunk in response.split():
        full_response += chunk + " "
        time.sleep(0.001)
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    return full_response, response


prompt = st.chat_input("arxiv_id OR arxiv_url @ Your question here")
# If the user has entered a prompt, chat with the assistant
if prompt:
    prompt = get_predefined_prompt(prompt)
    # index of first comma
    start_index = prompt.find("@")
    if start_index == -1:
        st.warning("Please provide the question with @")
        st.stop()

    arxiv_id = prompt[:start_index].strip()
    question = prompt[start_index + 1 :].strip()
    if not valid_arxiv_id(arxiv_id) and not valid_arxiv_url(arxiv_id):
        st.warning("Please provide a valid arxiv_id or arxiv_url")
        st.stop()

    if valid_arxiv_url(arxiv_id):
        arxiv_id = get_arxiv_id_from_url(arxiv_id)

    paper_content = get_paper_content(arxiv_id)
    paper_metadata = get_paper_metadata(arxiv_id)

    if not paper_metadata:
        st.error("Paper not found in the knowledge base. Please enter a valid arXiv ID or URL.")
        st.stop()

    if not paper_content:
        st.error("Paper content not found. Please try again later.")
        st.stop()

    user_prompt = f"[:green[{paper_metadata['title']}]]({build_url(arxiv_id)})\n{question}"

    # Add the user's prompt to the message history and display it
    st.session_state.page_states[session_id].add_message({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt, unsafe_allow_html=True)

    # Display the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # First just get response
        full_response, response = qa_paper(question, paper_content, message_placeholder)

        # Add the response and sources to the message history
        message = {"role": "assistant", "content": full_response}
        st.session_state.page_states[session_id].add_message(message)
