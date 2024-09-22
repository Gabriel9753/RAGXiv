import re
import streamlit as st
import os
import sys
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

root_dir = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(root_dir)

from src.scripts.db_manager import DBManager
import rag.memory as mem
from rag.chains import (
    stuff_chain,
    reduce_chain,
    reranker_chain,
    semantic_search_chain,
    hyde_chain,
    summarization_chain,
    paper_qa_chain,
)
from rag.utils import load_vectorstore, load_llm, load_embedding, build_runnable

load_dotenv()
COLLECTION_NAME = "arxiv_papers_RecursiveCharacterTextSplitter"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")


class PageState:
    def __init__(self, session_id, subpage_name):
        """
        Initialize a PageState object.

        Args:
            session_id (str): The unique identifier for the session.
            subpage_name (str): The name of the subpage.
        """
        self.subpage_name = subpage_name
        self.session_id = session_id
        self.messages = []
        self.memory = None
        self.runnable = None
        self.chain = None
        self.cur_model = None

    def get_messages(self):
        """
        Retrieve all messages stored in the PageState.

        Returns:
            list: A list of message dictionaries.
        """
        return self.messages

    def add_message(self, message):
        """
        Add a new message to the PageState.

        Args:
            message (dict): The message to be added.
        """
        self.messages.append(message)

    def clear_messages(self):
        """
        Clear all messages from the PageState.
        """
        self.messages = []

    def set_rag_components(self, chain, memory, runnable):
        """
        Set the RAG components for the PageState.

        Args:
            chain: The chain component.
            memory: The memory component.
            runnable: The runnable component.
        """
        self.chain = chain
        self.memory = memory
        self.runnable = runnable

    def get_rag_components(self):
        """
        Retrieve the RAG components from the PageState.

        Returns:
            tuple: A tuple containing the chain, memory, and runnable components.
        """
        return self.chain, self.memory, self.runnable

    def set_model(self, model):
        """
        Set the current model for the PageState.

        Args:
            model: The model to be set.
        """
        self.cur_model = model

    def get_model(self):
        """
        Retrieve the current model from the PageState.

        Returns:
            The current model.
        """
        return self.cur_model


@st.cache_resource
def get_retreiver():
    """
    Get a cached retriever instance.

    Returns:
        Retriever: An instance of the retriever.
    """
    retriever = load_vectorstore(QDRANT_URL, QDRANT_API_KEY).as_retriever()
    return retriever


def get_rag_components(_chain, _model):
    """
    Get RAG components based on the specified chain and model.

    Args:
        _chain (str): The type of chain to use.
        _model: The model to use.

    Returns:
        tuple: A tuple containing the chain, memory, and runnable components.

    Raises:
        ValueError: If an invalid chain type is provided.
    """
    rag_llm = load_llm(temp=0.3, _model=_model)
    rag_retriever = get_retreiver()
    memory = mem.Memory()

    chain = None
    if _chain == "stuffing":
        chain = stuff_chain(rag_llm=rag_llm, rag_retriever=rag_retriever)
    elif _chain == "reduction":
        reduce_llm = load_llm(temp=0.05, _model=_model)
        chain = reduce_chain(qa_llm=rag_llm, reduce_llm=reduce_llm, rag_retriever=rag_retriever)
    elif _chain == "reranking":
        chain = reranker_chain(rag_llm=rag_llm, rag_retriever=rag_retriever)
    elif _chain == "hyde":
        chain = hyde_chain(rag_llm=rag_llm, rag_retriever=rag_retriever)
    elif _chain == "semantic_search":
        chain = semantic_search_chain(rag_llm=rag_llm, rag_retriever=rag_retriever)
    elif _chain == "summarization":
        chain = summarization_chain(rag_llm=rag_llm)
    elif _chain == "paper_qa":
        chain = paper_qa_chain(rag_llm=rag_llm)
    else:
        raise ValueError(f"Invalid chain type: {_chain}")
    runnable = build_runnable(chain, memory)
    return chain, memory, runnable


@st.cache_resource
def get_db_manager():
    """
    Get a cached instance of the DBManager.

    Returns:
        DBManager: An instance of the DBManager.
    """
    return DBManager()


def display_previous_messages(session_id):
    """
    Display previous messages for a given session.

    Args:
        session_id (str): The unique identifier for the session.
    """
    for message in st.session_state.page_states[session_id].get_messages():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and st.session_state.get("show_sources", True):
                with st.expander("Sources", expanded=False):
                    for source in message["sources"]:
                        st.markdown(source)


def get_retreived_papers(response):
    """
    Extract information about retrieved papers from a response.

    Args:
        response (dict): The response containing retrieved papers.

    Returns:
        dict: A dictionary of retrieved papers with their metadata.
    """
    db_manager = get_db_manager()
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
    """
    Build a markdown-formatted list of used papers.

    Args:
        used_papers (dict): A dictionary of used papers.

    Returns:
        list: A list of markdown-formatted strings for each paper.
    """
    papers = list(used_papers.items())[:5]  # Limit to 5 papers
    sources_list = []

    for arxiv_id, paper in papers:
        title = paper.get("title", arxiv_id)
        pages = paper.get("pages")
        pages_str = ", ".join([f"p. {page}" for page in sorted(pages)])
        source = f"[{title}](https://arxiv.org/abs/{arxiv_id}) (:grey[{pages_str}])"
        sources_list.append(source)

    return sources_list


def get_references(arxiv_id):
    """
    Get references for a given arXiv ID.

    Args:
        arxiv_id (str): The arXiv ID of the paper.

    Returns:
        list: A list of references for the given paper.
    """
    db_manager = get_db_manager()
    references = db_manager.get_references_from_arxivid(arxiv_id)
    return references


def get_paper_metadata(arxiv_id):
    """
    Get metadata for a given arXiv ID.

    Args:
        arxiv_id (str): The arXiv ID of the paper.

    Returns:
        dict: Metadata of the paper.
    """
    db_manager = get_db_manager()
    metadata = db_manager.get_metadata_from_arxivid(arxiv_id)
    return metadata


def get_authors(arxiv_id):
    """
    Get authors for a given arXiv ID.

    Args:
        arxiv_id (str): The arXiv ID of the paper.

    Returns:
        list: A list of authors for the given paper.
    """
    db_manager = get_db_manager()
    authors = db_manager.get_authors_from_arxivid(arxiv_id)
    return authors


def scale_similarities(similarities):
    """
    Scale similarity values to range [0, 1].

    Args:
        similarities (dict): A dictionary of similarity values.

    Returns:
        dict: A dictionary of scaled similarity values.
    """
    min_sim = min(similarities.values())
    max_sim = max(similarities.values())
    if min_sim == max_sim:
        return {k: 1 for k in similarities}
    return {k: (v - min_sim) / (max_sim - min_sim) for k, v in similarities.items()}


def cosine_similarity(a, b):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        a (numpy.array): First vector.
        b (numpy.array): Second vector.

    Returns:
        float: The cosine similarity between vectors a and b.
    """
    return dot(a, b) / (norm(a) * norm(b))


def get_title_similarity_values(main_title, titles, do_scale=True):
    """
    Calculate similarity values between a main title and a list of titles.

    Args:
        main_title (str): The main title to compare against.
        titles (list): A list of titles to compare with the main title.
        do_scale (bool): Whether to scale the similarity values.

    Returns:
        dict: A dictionary of titles and their similarity values to the main title.
    """
    embedder = load_embedding()
    title_embeddings = embedder.embed_documents(titles)
    main_title_embedding = embedder.embed_query(main_title)
    # Calculate cosine similarity between main_title and all titles without embedder
    similarity_values = {
        title: cosine_similarity(main_title_embedding, title_embedding)
        for title, title_embedding in zip(titles, title_embeddings)
    }

    similarity_values = scale_similarities(similarity_values) if do_scale else similarity_values
    return similarity_values


@st.cache_resource
def get_qdrant_client():
    """
    Get a cached instance of the Qdrant client.

    Returns:
        tuple: A tuple containing the Qdrant client and collection name.
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collection_name = COLLECTION_NAME
    return client, collection_name


def get_paper_content(arxiv_id):
    """
    Retrieve the content of a paper given its arXiv ID.

    Args:
        arxiv_id (str): The arXiv ID of the paper.

    Returns:
        str: The content of the paper.
    """
    client, collection_name = get_qdrant_client()
    results = client.scroll(
        collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="metadata.arxiv_id", match=models.MatchValue(value=arxiv_id))],
        ),
        limit=99,
    )[0]
    content = " ".join([doc.payload["page_content"] for doc in results])
    return content


def get_similar_papers(prompt):
    """
    Find papers similar to a given prompt.

    Args:
        prompt (str): The prompt to find similar papers for.

    Returns:
        list: A list of similar papers with their similarity scores.
    """
    vectorstore = load_vectorstore(QDRANT_URL, QDRANT_API_KEY)
    # retriever = vectorstore.as_retriever()
    response = vectorstore.similarity_search_with_score(prompt, k=5)
    return response

def valid_arxiv_id(arxiv_id):
    """
    Check if the arxiv_id is valid.

    Args:
        arxiv_id (str): The arXiv ID to validate.

    Returns:
        bool: True if the arXiv ID is valid, False otherwise.

    Note:
        A valid arXiv ID should match the pattern: YYMM.NNNNN or YYMM.NNNNNvV
        where YY is the year, MM is the month, NNNNN is a 4 or 5 digit number,
        and V is an optional version number.
    """
    return bool(re.match(r"\d{4}\.\d{4,5}(v\d+)?", arxiv_id))


def valid_arxiv_url(arxiv_url):
    """
    Check if the arxiv_url is valid.

    Args:
        arxiv_url (str): The arXiv URL to validate.

    Returns:
        bool: True if the arXiv URL is valid, False otherwise.

    Note:
        A valid arXiv URL should start with either 'https://arxiv.org/abs/'
        or 'https://arxiv.org/pdf/' and end with a valid arXiv ID.
    """
    abs_url = "https://arxiv.org/abs/"
    pdf_url = "https://arxiv.org/pdf/"
    url_check = arxiv_url.startswith(abs_url) or arxiv_url.startswith(pdf_url)
    return url_check and valid_arxiv_id(arxiv_url.split("/")[-1])


def get_arxiv_id_from_url(arxiv_url):
    """
    Extract the arxiv_id from the arxiv_url.

    Args:
        arxiv_url (str): The arXiv URL to extract the ID from.

    Returns:
        str: The extracted arXiv ID.

    Note:
        This function assumes the arXiv ID is the last part of the URL path.
    """
    return arxiv_url.split("/")[-1]


def build_url(arxiv_id):
    """
    Build a full arXiv URL from an arXiv ID.

    Args:
        arxiv_id (str): The arXiv ID to build the URL for.

    Returns:
        str: The complete arXiv URL for the abstract page.

    Note:
        This function constructs a URL for the abstract page on arXiv.org.
    """
    return f"https://arxiv.org/abs/{arxiv_id}"


def get_predefined_prompt(prompt):
    """
    Get a predefined prompt based on the input.

    Args:
        prompt (str): The input prompt or predefined prompt code.

    Returns:
        str: The resolved prompt.
    """
    # some predefined prompts
    if prompt.lower() == "/p1":
        prompt = "What is the method 'Stuffing' used for in a RAG-based chatbot?"
    elif prompt.lower() == "/p2":
        prompt = "What is the method 'Reduction' used for in a RAG-based chatbot?"
    elif prompt.lower() == "/p3":
        prompt = "What is the method 'Reranking' used for in a RAG-based chatbot?"
    elif prompt.lower() == "/p4":
        prompt = "What is the method 'HyDE' used for in a RAG-based chatbot?"
    elif prompt.lower() == "/p5":
        prompt = "In the context of learning with distribution shift, how does the complexity of the function class G, which represents the nuisance function, affect the performance of the predictor when the shift in the marginal distribution of y is significantly smaller than the shift in the joint distribution of (x, y)?"
    elif prompt.lower() == "/p6":
        prompt = "In the context of federated learning, what are the key differences between the proposed method and differentially private federated learning (DPFL) in terms of their mechanisms, certification goals, and technical contributions?"
    # some predefined arxiv ids
    elif prompt.lower() == "/a1":
        prompt = "2312.03511"
    elif prompt.lower() == "/a2":
        prompt = "2305.01644"
    elif prompt.lower() == "/a3":
        prompt = "2404.16689"
    elif prompt.lower() == "/a4":
        # Attention is All You Need
        prompt = "1706.03762"
    elif prompt.lower() == "/a5":
        # DINO
        prompt = "2104.14294"
    # some predefined questions to specific paper
    elif prompt.lower() == "/q1":
        prompt = "1706.03762 @ How does the use of multiple layers in both the encoder and decoder contribute to the model’s ability to capture complex dependencies in long sequences?"
    elif prompt.lower() == "/q2":
        prompt = "1706.03762 @ What is the effect of the number of attention heads in the model on the performance of the model?"
    elif prompt.lower() == "/q3":
        prompt = "1706.03762 @ Why are additive attention mechanisms less efficient than dot-product attention, particularly for larger dimensions?"
    else:
        prompt = prompt
    return prompt
