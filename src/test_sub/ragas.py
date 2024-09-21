import os

import ragas
import ragas.run_config

from datasets import load_dataset
from langchain_openai import ChatOpenAI
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
# from ragas.testset.generator import TestsetGenerator
# from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_community.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI

if __name__ == "__main__":
    ds = load_dataset("MarkrAI/AutoRAG-evaluation-2024-LLM-paper-v1", "qa")


    # In[8]:


    ds =ds["train"].select(range(10)).to_pandas()
    ds.rename(columns={"query": "question", "generation_gt": "ground_truth"}, inplace=True)
    ds = ds[["question", "ground_truth"]]
    ds


    # In[9]:





    llm = ChatOpenAI(openai_api_base="http://localhost:5000/v1", openai_api_key="lm-studio")
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/allenai-specter", model_kwargs={"device": "cpu"})

    # generator = TestsetGenerator.from_langchain(
    #     generator_llm=llm,
    #     critic_llm=llm,
    #     embeddings=emb,
    # )

    # testset = generator.generate_with_langchain_docs(chunks, test_size=1, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}, raise_exceptions=False)


    # In[11]:




    loader = WebBaseLoader("https://blog.langchain.dev/langgraph-multi-agent-workflows/")
    docs = loader.load()






    def remove_unessesary_lines(docs):
        lines = ""
        for doc in docs:
                lines += doc.page_content
        new_lines = lines.split("\n")
        stripped_lines = [line.strip() for line in new_lines]
        non_empty_lines = [line for line in stripped_lines if line]
        cleaned_content = "".join(non_empty_lines)
        return cleaned_content

    cleaned_content = remove_unessesary_lines(docs)

    #converting to a Document format for embedding
    new_doc = [Document(page_content=cleaned_content,metadata =docs[0].metadata)]
        


    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(new_doc)
        


    # In[13]:


    from langchain_community.vectorstores import FAISS


    db = FAISS.from_documents(chunks, emb)
    retriever = db.as_retriever()


    # In[14]:


    from langchain_core.prompts import PromptTemplate

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context","question"]
    )


    # In[18]:




    # This controls how each document will be formatted. Specifically,
    # it will be passed to `format_document` - see that function for more
    # details.
    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )
    document_variable_name = "context"
    # The prompt here should take as an input variable the
    # `document_variable_name`
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name
    )



    # In[46]:


    from datasets import Dataset

    # Prepare the questions and ground truths
    questions = ds["question"].to_list()
    ground_truth = ds["ground_truth"].to_list()
    if isinstance(ground_truth[0], list):
        ground_truth = [gt[0] for gt in ground_truth]

    # Initialize an empty data dictionary for the new dataset
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": ground_truth}

    # Simulate RAG process: querying and retrieving documents
    for query in questions:
        # Mock RAG chain and retriever responses (replace with actual RAG chain and retriever)
        # mock_answer = "This is a mock answer for query: " + query  # Simulate answer from RAG
        # mock_context = ["This is a context for the query: " + query]  # Simulate context from retriever
        docs = retriever.invoke(query, top_k=4)
        answer = chain.invoke({"question": query, "input_documents": docs})
        context = [doc.page_content for doc in docs]
        
        # Append the results to the data dictionary
        data["question"].append(query)
        data["answer"].append(answer["output_text"])
        data["contexts"].append(context)

    # Create a dataset from the dictionary
    dataset = Dataset.from_dict(data)


    # In[50]:


    dataset = dataset.to_pandas()
    dataset["ground_truth"] = [gt[0] for gt in dataset["ground_truth"]]
    dataset


    # In[54]:




    run_config = ragas.run_config.RunConfig(max_workers=1, timeout=15)

    os.environ["OPENAI_API_KEY"] = "test"
    result = ragas.evaluate(
        dataset = Dataset.from_pandas(dataset),
        metrics=[
            ragas.metrics.context_precision,
            ragas.metrics.context_recall,
            ragas.metrics.faithfulness,
            ragas.metrics.answer_relevancy,
        ],
        llm=llm,
        embeddings=emb,
    )


    # In[31]:


    df = result.to_pandas()
    df


    # In[32]:


    df.isnull().sum()


    # In[ ]:





    # In[30]:


    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    df = result.to_pandas()

    heatmap_data = df[['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']]

    cmap = LinearSegmentedColormap.from_list('green_red', ['red', 'green'])

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=.5, cmap=cmap)

    plt.yticks(ticks=range(len(df['question'])), labels=df['question'], rotation=0)

    plt.show()


    # ### Add LangFuse

    # In[ ]:


    # from langfuse import Langfuse

    # langfuse = Langfuse(
    #   secret_key="sk-lf-8be80c67-4187-4e43-9d01-544195dc9f03",
    #   public_key="pk-lf-d7653f64-8086-4365-b05c-865ead3478a3",
    #   host="http://localhost:3000"
    # )


    # # In[ ]:


    # trace = langfuse.trace(
    #     name = "eval",
    #     user_id = "eval_user",
    #     metadata = {
    #         "email": "prod@company.com",
    #     },
    #     tags = ["evaluation"]
    # )


    # # In[ ]:


    # df = result.to_pandas()


    # # In[ ]:


    # for _, row in df.iterrows():
    #     for metric_name in ["faithfulness", "answer_relevancy", "context_recall"]:
    #         langfuse.score(
    #             name=metric_name,
    #             value=row[metric_name],
    #             trace_id=trace.id
    #         )

