def format_docs(docs):
    """Formats the documents for prompt generation."""
    return "\n\n".join([d.page_content for d in docs])
