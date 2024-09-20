import os
import uuid

import yaml
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_qdrant import QdrantVectorStore

from langfuse.decorators import langfuse_context, observe
from langfuse.callback import CallbackHandler

import config
import utils

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

langfuse_handler = CallbackHandler(session_id="conversation_chain")

def get_embedding_function():
    """Returns the embedding function based on the configuration."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME,
                                 model_kwargs={"device": "cpu"})


def initialize_retriever():
    """
    Initialize the retriever using the HuggingFace embeddings and Qdrant vectorstore.
    Returns:
        Retriever: A retriever for retrieving relevant documents.
    """
    embedding_function = get_embedding_function()

    vs = QdrantVectorStore.from_existing_collection(
        collection_name=config.COLLECTION_NAME,
        embedding=embedding_function,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    return vs.as_retriever()


def initialize_llm():
    """Initialize and return the LLM based on the configured type."""
    if config.LLM_TYPE == "lm-studio":
        return ChatOpenAI(openai_api_base="http://localhost:5000/v1", openai_api_key="lm-studio")
    elif config.LLM_TYPE == "ollama":
        return OllamaLLM(model="llama3.1:8b", temperature=0.3)
    else:
        raise NotImplementedError(f"LLM type {config.LLM_TYPE} is not supported yet.")


def build_prompt_templates():
    """Create and return the prompt templates."""
    reformulate_system_prompt = (
        "Given a chat history and the latest user question, \
        reformulate the question to be understood independently."
    )
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. \
        Use the provided context to answer the question concisely.\n\n \
        Context: {context}"
    )

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [("system", reformulate_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )

    return contextualize_prompt, qa_prompt


def build_chain():
    """Create the RAG chain using the retriever, prompts, and LLM."""
    llm = initialize_llm()
    retriever = initialize_retriever()

    contextualize_prompt, qa_prompt = build_prompt_templates()

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


class Memory:
    def __init__(self):
        self.store = {}

    def get(self, session_id):
        """Retrieve session data by session_id."""
        return self.store.get(session_id)

    def set(self, session_id, value):
        """Store session data by session_id."""
        self.store[session_id] = value

    def get_session_history(self, session_id):
        """Retrieve or initialize chat history for a session."""
        print("SessionID: ", session_id)
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()

        print("Session History: ", self.store[session_id], type(self.store[session_id]))

        return self.store[session_id]

    def clear(self):
        """Clear the memory."""
        self.store = {}

@observe(name="build_runnable()", as_type="generation")
def build_runnable(rag_chain, memory, keys: dict = None):
    """Build a runnable with message history"""

    if keys is None:
        keys = {"input_messages_key": "input", "history_messages_key": "chat_history", "output_messages_key": "answer"}

    return RunnableWithMessageHistory(
        rag_chain,
        memory.get_session_history,
        input_messages_key=keys["input_messages_key"],
        history_messages_key=keys["history_messages_key"],
        output_messages_key=keys["output_messages_key"],
    )

@observe(name="chat()", as_type="generation")
def chat(rag, input_message, session_id, stream=False):
    """Chat with the model using the RAG chain."""
    # langfuse_handler = langfuse_context.get_current_langchain_handler()

    response = rag.invoke(
        {"input": input_message},
        config={"configurable": {"session_id": session_id}, "callbacks": [langfuse_handler]},
    )

    # langfuse_context.update_current_observation(
    #     input=input_message, output=response["answer"], session_id=session_id
    # )
    return response

@observe(name="get_similar_papers()", as_type="retrieval")
def get_similar_papers(vectorstore, query, k=5):
    """Get similar papers based on a query"""
    results = vectorstore.similarity_search(query, k=k)
    return [{"title": doc.metadata.get("title"), "arxiv_id": doc.metadata.get("arxiv_id")} for doc in results]

@observe(name="get_paper_questions()", as_type="retrieval")
def get_paper_questions(vectorstore, arxiv_id, k=5):
    """Get relevant questions for a specific paper"""
    results = vectorstore.similarity_search(f"arxiv_id:{arxiv_id}", k=k)
    return [doc.page_content for doc in results if doc.metadata.get("arxiv_id") == arxiv_id]



def summarize(llm, paper):
    from langchain.chains.summarize import load_summarize_chain
    #TODO FETCH PAPER
    chain = load_summarize_chain(llm, chain_type="stuff", verbose=True)
    chain.run({"input": paper})



if __name__ == "__main__":
    from langfuse import Langfuse

    langfuse = Langfuse()

    print("Langfuse available: ",langfuse.auth_check())
    # Initialize components
    chain = build_chain()
    yaml_path = "app/questions.yaml"

    with open(yaml_path, "r", encoding="utf-8") as f:
        questions = yaml.load(f, Loader=yaml.FullLoader)["questions"]
    memory = Memory()
    runnable = build_runnable(chain, memory)
    session_id = str(uuid.uuid4())
    for i, question in enumerate(questions, 1):
        print(f"##################\n### Question {i} ###\n##################")

        q = question["question"]
        a = question["answer"]
        pdf = question["paper"]

        # Chat with the model
        output = chat(runnable, q, session_id=session_id)
        # print(
        #     f'~~~ Question ~~~\n{q}\n\n~~~ Output ~~~\n{output["answer"]}\n\n~~~ "Correct" Answer ~~~\n{a}\n\n~~~ Paper ~~~\n{pdf}\n'
        # )
        print(output)



