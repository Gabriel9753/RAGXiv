from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from operator import itemgetter

CHROMADIR = r"chroma/"

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def main():
    template = """Answer the question in the question's language based on your knowledge and the following context:
        {context}

        Question: {question}
    """

    condense_question = PromptTemplate.from_template(template="Condense the following question: {question}")

    llm = ChatOpenAI(openai_api_base="http://localhost:5000/v1", openai_api_key="lm-studio")
    prompt = ChatPromptTemplate.from_template(template=template)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/allenai-specter")
    vectorstore = Chroma(persist_directory=CHROMADIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    chain = (
        RunnablePassthrough.assign(source_documents=itemgetter("question") | retriever)
        | RunnablePassthrough.assign(context=lambda inputs: format_docs(inputs["source_documents"]) if inputs["source_documents"] else "")
        | RunnablePassthrough.assign(prompt=prompt)
        | RunnablePassthrough.assign(response=lambda inputs: llm(inputs["prompt"].messages))
    )
    INPUT = "Wie öffnet man die Vordertür im Notfall?"

    output = chain.invoke({"question": INPUT})

if __name__ == "__main__":
    main()