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

PAPERQA_TEMPLATE = (
    "You are an assistant for question-answering tasks. \
    Use the provided paper content to answer the question concisely.\n\n \
    Paper Content: \n\n{context}"
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

PAPERQA_PROMPT = ChatPromptTemplate.from_messages(
    [("system", PAPERQA_TEMPLATE),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")]
)

SUMMARIZATION_TEMPLATE = """
    Summarize the following academic paper from arXiv, focusing on the key points and contributions. Include a brief overview of the problem the paper addresses, the methods used, the main findings, and any conclusions or implications. Aim to condense the content while maintaining accuracy and clarity. Use clear and concise language, avoiding technical jargon when possible.

    Here is the paper content:

    {input}
    """

SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SUMMARIZATION_TEMPLATE),
     MessagesPlaceholder("chat_history"),
     ("human", "{input}")])


HYDE_TEMPLATE = """
    Given the chat history and the latest user question, answer the question by providing a
    hypothetical document that represents the user's query. Use the chat history to provide context.
    """

HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", HYDE_TEMPLATE),
     MessagesPlaceholder("chat_history"),
     ("human", "{input}")]
)