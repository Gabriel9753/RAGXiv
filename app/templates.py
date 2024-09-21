from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


CONTEXTUALIZE_TEMPLATE = (
    "Given a chat history and the latest user question, \
    reformulate the question to be understood independently."
)
QA_TEMPLATE = (
    "You are an assistant for question-answering tasks. \
    Use the provided context to answer the question concisely.\n\n \
    Context: {context}"
)

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", CONTEXTUALIZE_TEMPLATE), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [("system", QA_TEMPLATE), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)


CONTEXTUALIZE_TEMPLATE = (
    "Given a chat history and the latest user question, \
    reformulate the question to be understood independently."
)
QA_TEMPLATE = (
    "You are an assistant for question-answering tasks. \
    Use the provided context to answer the question concisely.\n\n \
    Context: {context}"
)

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", CONTEXTUALIZE_TEMPLATE),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [("system", QA_TEMPLATE),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")]
)

COMBINE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", "Provide a ombine the following documents into a single document: {context}."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")]
)

COLLAPSE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", "Provide the context in a collapsed, precise form: \n\n{context}."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")]

)

SEMANTIC_SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [("system", "Retrieve documents related to the following query. As an answer, provide a short introduction to the retrieved documents: {context}."),
        ("human", "Query: {input}")]
)