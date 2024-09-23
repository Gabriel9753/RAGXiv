from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -----------------------------------------------------
#                      Templates
# -----------------------------------------------------

# Reformulate user questions in the context of previous conversations
CONTEXTUALIZE_TEMPLATE = """
    Given the ongoing conversation and the most recent user question, reformulate the question
    so that it can be understood without the chat history.
"""

# Use concise language for question-answering tasks
QA_TEMPLATE = """
    You are an assistant designed to answer questions based on the provided context.
    Answer the user's question concisely and accurately using the context below.
    Be short and to the point.

    Context: {context}
"""

# Paper-specific question-answering
PAPERQA_TEMPLATE = """
    You are an assistant for answering questions based on academic paper content.
    Use the paper information below to answer the question clearly and concisely.
    Be short and to the point.

    Paper Content: {context}
"""

# Academic paper summarization prompt
SUMMARIZATION_TEMPLATE = """
    Summarize the following academic paper from arXiv, highlighting the key points and contributions.
    Include an overview of the problem addressed, methods used, main findings, and conclusions.
    The summary should be clear, concise, and free of excessive technical jargon.

    Paper Content: {input}
"""

# Hypothetical document creation for question answering
HYDE_TEMPLATE = """
    Based on the conversation history and the latest question, generate a hypothetical document
    that would represent an ideal response to the user's query, using the chat history for context.
"""

# -----------------------------------------------------
#                       Prompts
# -----------------------------------------------------
# Reformulate a user question based on chat history
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# General question-answering prompt
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# Document merging prompt
COMBINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Combine the following documents into a cohesive summary: {context}."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# Collapsing context into a shorter form
COLLAPSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Summarize the following context into a concise, clear form:\n\n{context}."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# Semantic search introduction
SEMANTIC_SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Retrieve and summarize the documents related to the following query. Provide a brief introduction to the relevant documents:\n\n{context}.",
        ),
        ("human", "Query: {input}")
    ]
)

# Paper-specific question-answering with academic content
PAPERQA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", PAPERQA_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)


# Academic paper summarization prompt
SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SUMMARIZATION_TEMPLATE), ("human", "{input}")]
)


# Hypothetical document creation for question answering
HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", HYDE_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
