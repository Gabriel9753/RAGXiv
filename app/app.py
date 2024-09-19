import streamlit as st
import uuid
import time
from rag import initialize, Memory, build_runnable, chat, get_similar_papers, get_paper_questions
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.scripts.db_manager import DBManager

# Set page config first
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

# Include custom JavaScript
st.components.v1.html(
    """
    <script src="app/static/streamlit_app.js"></script>
    """,
    height=0,
)

@st.cache_resource
def get_rag_components():
    components = initialize()
    rag_chain = components["chain"]
    vectorstore = components["vectorstore"]
    memory = Memory()
    runnable = build_runnable(rag_chain, memory)
    return components, rag_chain, vectorstore, memory, runnable

@st.cache_resource
def get_db_manager():
    return DBManager()

# Use the cached function to get the components
components, rag_chain, vectorstore, memory, runnable = get_rag_components()
db_manager = get_db_manager()

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.title("📚 RAG Chatbot")

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This chatbot uses Retrieval-Augmented Generation (RAG) to provide "
    "informative responses based on a knowledge base of academic papers."
)

# Command buttons
# TODO: Commands not working
st.sidebar.header("Quick Commands")
st.sidebar.button("Get Similar Papers", on_click=lambda: st.components.v1.html(
    f"""<script>insertCommand('/similar ')</script>""",
    height=0
))
st.sidebar.button("Get Paper Questions", on_click=lambda: st.components.v1.html(
    f"""<script>insertCommand('/q ')</script>""",
    height=0
))

# Main chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def prompt_similar(prompt):
    similar_papers = get_similar_papers(vectorstore, prompt[8:])
    full_response = "Here are similar papers:\n\n" + "\n".join([f"- [{paper['title']}](https://arxiv.org/abs/{paper['arxiv_id']})" for paper in similar_papers])
    return full_response

def prompt_question(prompt):
    """Get answers to questions for a specific paper"""
    arxiv_id = prompt.split()[1]
    questions = get_paper_questions(vectorstore, arxiv_id)
    full_response = f"Here are relevant questions for paper {arxiv_id}:\n\n" + "\n".join([f"- {q}" for q in questions])
    return full_response

def prompt_db(prompt, message_placeholder):
    """Just a test function to interact with the database"""
    full_response = ""
    _input = prompt.split()[1:]
    _input = " ".join(_input)
    response = chat(runnable, _input)

    # Simulate stream of response with milliseconds delay
    for chunk in response["answer"].split():
        full_response += chunk + " "
        time.sleep(0.01)
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    used_papers = set()
    for doc in response["context"]:
        arxiv_id = doc.metadata.get("arxiv_id")
        if arxiv_id:
            used_papers.add(str(arxiv_id))
    metadata = [db_manager.get_metadata_from_arxivid(arxiv_id) for arxiv_id in used_papers]
    authors = [db_manager.get_authors_from_arxivid(arxiv_id)[:2] for arxiv_id in used_papers]
    references = [db_manager.get_references_from_arxivid(arxiv_id)[:2] for arxiv_id in used_papers]

    full_response += "\n\n*Sources:* " + ", ".join(used_papers)
    for meta in metadata:
        for key, value in meta.items():
            full_response += f"\n{key}: {value}"
        full_response += "\n"


    for paper_authors, aid in zip(authors, used_papers):
        full_response += f"\nAuthors for paper {aid}:"
        for author in paper_authors:
            for key, value in author.items():
                full_response += f"\n{key}: {value}"
            full_response += "\n"

    for paper_references, aid in zip(references, used_papers):
        full_response += f"\nReferences for paper {aid}:"
        for reference in paper_references:
            for key, value in reference.items():
                full_response += f"\n{key}: {value}"
            full_response += "\n"

    return full_response


def normal_chat(prompt, message_placeholder):
    full_response = ""
    response = chat(runnable, prompt)
    # Simulate stream of response with milliseconds delay
    for chunk in response["answer"].split():
        full_response += chunk + " "
        time.sleep(0.01)
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)

    # Extract used papers and pages
    used_papers = set()
    for doc in response["context"]:
        title = doc.metadata.get("title")
        arxiv_id = doc.metadata.get("arxiv_id")
        page = doc.metadata.get("page")
        if title and arxiv_id:
            used_papers.add(f"[{title}](https://arxiv.org/abs/{arxiv_id}) (p. {page})")

    if used_papers:
        full_response += "\n\n*Sources:* " + ", ".join(used_papers)

    return full_response

prompt = st.chat_input("Ask me anything about the papers in the knowledge base")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if prompt.startswith("/similar"):
            full_response = prompt_similar(prompt)
        elif prompt.startswith("/q"):
            full_response = prompt_question(prompt)
        elif prompt.startswith("/testdb"):
            full_response = prompt_db(prompt, message_placeholder)
        else:
            full_response = normal_chat(prompt, message_placeholder)

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    memory.set(st.session_state.session_id, st.session_state.messages)

# Add a button to clear the chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    memory.set(st.session_state.session_id, None)
    st.rerun()

# Display the session ID in the sidebar
st.sidebar.text(f"Session ID: {st.session_state.session_id}")