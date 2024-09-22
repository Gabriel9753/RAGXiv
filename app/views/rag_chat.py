import time
import streamlit as st
from streamlit_utils import get_rag_components, PageState, get_retreived_papers, build_used_papers_markdown, display_previous_messages, get_predefined_prompt
from utils import chat
import os
import hashlib

# create an unique session id for this subpage
cur_file = os.path.basename(__file__)
page_name = os.path.basename(__file__).split(".")[0]
session_id = hashlib.md5(page_name.encode()).hexdigest()

# Initialize session state
if session_id not in st.session_state.page_states:
    st.session_state.page_states[session_id] = PageState(session_id, page_name)
    get_rag_components.clear()
if "rag_method" not in st.session_state:
    st.session_state.rag_method = "stuffing"

# Get the RAG components (specific chain), depending on the method chosen
print(f"SELECTED LLM: {st.session_state.llm}")

st.title(":rainbow[RAG Chat]")
st.markdown("Welcome to the RAG Chat! Ask me anything about the papers in the knowledge base.")
st.markdown("In the sidebar, you can select the RAG method to use and clear the chat history.")
st.markdown("---")

chain, memory, runnable = get_rag_components(_chain=st.session_state.rag_method)

with st.sidebar:
    st.markdown("### RAG Methods")
    # TODO: Add missing chains
    rag_method = st.selectbox("Which method to use?", ("Stuffing", "Reduction", "Reranking", "HyDE")).lower()
    # If the method has changed, clear the chat history and update the RAG components
    if st.session_state.rag_method != rag_method:
        st.session_state.page_states[session_id].clear_messages()
        get_rag_components.clear()
        chain, memory, runnable = get_rag_components(_chain=rag_method)
        st.session_state.rag_method = rag_method

# If the user clicks the "Clear chat history" button, clear the chat history
def clear_chat_history():
    st.session_state.page_states[session_id].clear_messages()
    memory.clear()

if st.sidebar.button("Clear chat history"):
    clear_chat_history()

# Display the chat history for the current session
display_previous_messages(session_id)

def normal_chat(prompt, message_placeholder):
    full_response = ""
    response = chat(runnable, prompt, session_id, trace_name="summarization")
    for chunk in response["answer"].split():
        full_response += chunk + " "
        time.sleep(0.001)
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    return full_response, response

prompt = st.chat_input("Ask me anything about the papers in the knowledge base")
# If the user has entered a prompt, chat with the assistant
if prompt:
    if prompt.lower() == "/clear":
        clear_chat_history()
    else:
        prompt = get_predefined_prompt(prompt)
        # Add the user's prompt to the message history and display it
        st.session_state.page_states[session_id].add_message({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display the assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            # First just get response
            full_response, response = normal_chat(prompt, message_placeholder)
            # In response the used papers are stored, get them and build markdown
            if st.session_state.rag_method == "stuffing":
                used_papers = get_retreived_papers(response)
                sources_list = build_used_papers_markdown(used_papers)
            elif st.session_state.rag_method == "reduction":
                sources_list = [response.get("context", "")]
            elif st.session_state.rag_method == "reranking":
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
