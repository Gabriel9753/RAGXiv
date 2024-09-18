# Import necessary libraries
import config
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
import yaml
import uuid


def retriever():
    """
    Initialize the retriever using the HuggingFace embeddings and Chroma for persistence.

    Returns:
        Retriever: A retriever for retrieving relevant documents.
    """
    embedding_function = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})

    # Initialize the vectorstore using Chroma for persistence
    vectorstore = Chroma(persist_directory=config.CHROMADIR, embedding_function=embedding_function)

    return vectorstore.as_retriever()


contextualize_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise. "
    "If the question requires specific information/numbers/metrics from the paper, "
    "then use that information to answer the question and cite the source. "
    "\n\n"
    "{context}"
    "\n\n"
    "If you can't find the answer, use your own knowledge to "
    "provide a good answer."
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def initialize():
    if config.LLM_TYPE == "llm-studio":
        llm = ChatOpenAI(openai_api_base="http://localhost:5000/v1", openai_api_key="lm-studio")
    elif config.LLM_TYPE == "ollama":
        llm = OllamaLLM(model="llama3.1:8b", temperature=0.3)
    else:
        raise ValueError(f"Invalid LLM_TYPE: {config.LLM_TYPE}")

    history_aware_retriever = create_history_aware_retriever(llm, retriever(), contextualize_prompt)

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Return all the components for use
    return {"llm": llm, "retriever": history_aware_retriever, "chain": rag_chain}


class Memory:
    def __init__(self):
        self.store = {}
        # TODO: Add persistent storage

    def get(self, session_id):
        """Getter"""
        return self.store.get(session_id)

    def set(self, session_id, value):
        """Setter"""
        self.store[session_id] = value

    def get_session_history(self, session_id):
        """Getter for chat history"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]


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


def chat(rag, input_message):
    """Chat with the model"""
    # Get the session history
    session_id = str(uuid.uuid4())
    output_message = rag.invoke({"input": input_message}, config={"configurable": {"session_id": session_id}})

    return output_message


if __name__ == "__main__":
    # Initialize components
    llm, history_retriever, chain = initialize().values()
    yaml_path = "app/questions.yaml"

    with open(yaml_path, "r") as f:
        questions = yaml.load(f, Loader=yaml.FullLoader)["questions"]

    for i, question in enumerate(questions, 1):
        print(f"##################\n### Question {i} ###\n##################")
        memory = Memory()
        runnable = build_runnable(chain, memory)
        q = question["question"]
        a = question["answer"]
        pdf = question["paper"]

        # Chat with the model
        output = chat(runnable, q)["answer"]
        print(
            f'~~~ Question ~~~\n{q}\n\n~~~ Output ~~~\n{output}\n\n~~~ "Correct" Answer ~~~\n{a}\n\n~~~ Paper ~~~\n{pdf}\n'
        )

    # Retrieval of inputs
    retr = retriever()
    # print(retr.vectorstore.similarity_search(input1))
    # print(retr.invoke(input1))
