import os
from operator import itemgetter
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

import config

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# Load templates for question and answer prompts
QUESTION_TEMPLATE = """
You are an intelligent assistant who can handle two types of queries: scientific questions related to research papers and general conversational questions.
For scientific queries:
- If the user’s question is related to a scientific paper, ask for more specific details, such as the name of the paper, field of study, or section to clarify.
For general conversation:
- If the question is conversational, feel free to answer informally or continue the conversation.
The user's input is: "{question}"
If this is a scientific question about a paper, respond by summarizing the question and identifying relevant sections or terms. If this is a general question, respond naturally.
Chat history for context: {chat_history}
"""
ANSWER_TEMPLATE = """
You are a helpful assistant. Respond to the following query by providing two different kinds of responses:

1. **For scientific questions** (about papers or research):
   - Provide a brief, accurate summary of relevant research findings, methods, or data.
   - Reference key points from any documents or papers that were retrieved.
   - Offer clarifications if needed and suggest where more information can be found in the document.

2. **For general conversation**:
   - Respond naturally and engage the user with informative or friendly answers, based on the conversation flow.

User's question: "{question}"
Relevant documents (if available): {context}
Chat history for context: {chat_history}

Answer:

"""

# Initialize the model (using your local LLM endpoint)
llm = ChatOpenAI(
    temperature=0.3,
    streaming=True,
    openai_api_base="http://localhost:5000/v1",
    openai_api_key="lm-studio"
)

# Initialize Hugging Face embeddings and Chroma vector store
embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

vs = QdrantVectorStore.from_existing_collection(
    collection_name=config.COLLECTION_NAME,
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
retriever = vs.as_retriever()



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
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()


        return self.store[session_id]

    def clear(self):
        """Clear the memory."""
        self.store = {}



contextualize_prompt = PromptTemplate.from_template(QUESTION_TEMPLATE)
answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

# Optimized standalone question generation and memory integration

memory = Memory()


# Map
map_template = "Write a concise summary of the following: {docs}."
map_prompt = ChatPromptTemplate([("human", map_template)])
map_chain = LLMChain(llm=llm, prompt=map_prompt)


# Reduce
reduce_template = """
The following is a set of summaries:
{docs}
Take these and distill it into a final summary.
"""
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)


# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=1000,
)

# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

if __name__ == "__main__":
    # Get user input

    user_question = "What is ChatGPT?"

    # # Execute the chain
    session_id = str(uuid.uuid4())

    retrieved_docs = retriever.invoke(user_question)
    print("Retrieved docs: ", retrieved_docs)

    result = map_reduce_chain.invoke(retrieved_docs)

    print(result["output_text"])
