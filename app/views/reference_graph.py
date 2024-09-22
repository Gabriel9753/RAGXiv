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

# If the user clicks the "Clear chat history" button, clear the chat history
if st.sidebar.button("Clear chat history"):
    st.session_state.page_states[session_id].clear_messages()

# Display the chat history for the current session
for message in st.session_state.page_states[session_id].get_messages():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "graph" in message:
            with st.expander("Graph", expanded=False):
                nodes, edges, config = message["graph"]
                agraph(nodes=nodes, edges=edges, config=config)


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


def build_graph(main_paper_title, arxiv_id, similarities):
    """Build the graph from the similarities."""
    nodes = []
    edges = []

    # Add the main paper as a node
    nodes.append(Node(id=arxiv_id, title=main_paper_title, size=40, color="#FF0000"))  # Red color for the main paper

    # Add similar papers as nodes and create edges
    for title, similarity in similarities.items():
        node_size = 5 + similarity * 20  # Scale node size based on similarity
        nodes.append(Node(id=arxiv_id + title, title=title, size=node_size))
        edge_width = 5 + similarity * 5  # Scale edge width based on similarity
        edges.append(Edge(source=title, target="main", width=edge_width))

    return nodes, edges


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

    references = get_references(arxiv_id)
    main_paper_title = get_paper_metadata(arxiv_id).get("title", arxiv_id)
    other_titles = [ref["title"] for ref in references]
    similarities = get_title_similarity_values(main_paper_title, other_titles)
    similarities = {title: similarity for title, similarity in similarities.items() if similarity > 0.3}

    # Display the graph
    with st.chat_message("assistant"):
        nodes, edges = build_graph(main_paper_title, arxiv_id, similarities)

        config = Config(
            width=1400,
            height=520,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=True,
        )

        st.markdown("### Reference Graph")
        agraph(nodes=nodes, edges=edges, config=config)

        # Add the response to the message history
        message = {"role": "assistant", "content": "Graph and references displayed.", "graph": (nodes, edges, config)}
        st.session_state.page_states[session_id].add_message(message)
