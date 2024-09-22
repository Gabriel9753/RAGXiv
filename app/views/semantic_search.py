from collections import defaultdict
import time
import numpy as np
import streamlit as st
from streamlit_utils import get_rag_components, PageState, get_retreived_papers, build_used_papers_markdown, display_previous_messages, get_predefined_prompt, get_similar_papers, get_paper_metadata, get_title_similarity_values
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

st.title(":rainbow[Sematic Search]")
st.markdown("Welcome to the Semantic Search! Just type in some context and I will find similar papers for you.")
st.markdown("---")

# If the user clicks the "Clear chat history" button, clear the chat history
if st.sidebar.button("Clear chat history"):
    st.session_state.page_states[session_id].clear_messages()

# Display the chat history for the current session
display_previous_messages(session_id)

def get_similar(prompt):
    similar_papers = get_similar_papers(prompt)
    return similar_papers

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
        similar_papers = get_similar(prompt)

        # used_papers = {p[0].metadata["arxiv_id"]: p[1] for p in similar_papers}
        used_papers = defaultdict(list)
        for p in similar_papers:
            used_papers[p[0].metadata["arxiv_id"]].append(p[1])
        used_papers_scores = {p: round(np.mean(sims), 2) for p, sims in used_papers.items()}
        used_papers_titles = {p: get_paper_metadata(p)["title"] for p in used_papers}
        used_papers = "\n".join([f"- [{used_papers_titles[p]}](https://arxiv.org/abs/{p}) (Similarity: {used_papers_scores[p]})" for p in used_papers])
        sim_paper_md = f"Here are some papers that are similar to the context you provided: \n{used_papers}"

        message_placeholder.markdown(sim_paper_md)

        # Add the response and sources to the message history
        message = {"role": "assistant", "content": sim_paper_md}
        st.session_state.page_states[session_id].add_message(message)
