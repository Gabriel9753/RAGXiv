from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config as cfg
from memory import ChatbotHistory, ChatSessionManager


class Retriever:
    """Retrieves documents using embeddings and Chroma vectorstore."""
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDING_MODEL_NAME)
        self.vectorstore = Chroma(persist_directory=cfg.CHROMADIR, embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever()

    def retrieve(self, query: str):
        """Retrieve documents based on a query using semantic search."""
        return self.retriever.invoke(input=query)


class RAGChain:
    """Implements Retrieval Augmented Generation using memory and retrieval."""
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.llm = self.initialize_llm()
        self.session_manager = ChatSessionManager()

    def initialize_llm(self) -> ChatOpenAI:
        """Initializes the LLM with API keys."""
        return ChatOpenAI(
            openai_api_base=cfg.OPENAI_API_BASE,
            openai_api_key=cfg.OPENAI_API_KEY
        )

    def create_prompt(self, query: str) -> ChatPromptTemplate:
        """Creates the prompt for the LLM, including history and the user query."""
        system_prompt = "You are a helpful assistant and professional scientist."
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", query),
            ]
        )

    def handle_query(self, session_id: str, query: str):
        """Handles a user query, combining history and retrieval."""
        # Retrieve documents
        retrieved_docs = self.retriever.retrieve(query)

        # Get session history
        session_history = self.session_manager.get_session_history(session_id)

        # Create prompt with history and query
        prompt = self.create_prompt(query)

        # Run prompt through LLM
        chain = prompt | self.llm
        response = chain.invoke({"history": session_history.get_history(), "question": query})

        # Add interaction to session history
        session_history.add_interaction(query, response)

        return response

