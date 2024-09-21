
# def init_langfuse():
#     from langfuse import Langfuse
#     langfuse = Langfuse()
#     print("Langfuse available: ",langfuse.auth_check())
#     from langfuse.decorators import langfuse_context, observe
#     from langfuse.callback import CallbackHandler
#     langfuse_handler = CallbackHandler(session_id="conversation_chain")
#     return langfuse


if __name__ == "__main__":
    llm = utils.load_llm()
    vs = utils.load_vectorstore(QDRANT_URL, QDRANT_API_KEY)
    retriever = vs.as_retriever()
    memory = Memory()

    runnable = build(llm, retriever, memory)

    yaml_path = "app/questions.yaml"

    with open(yaml_path, "r", encoding="utf-8") as f:
        questions = yaml.load(f, Loader=yaml.FullLoader)["questions"]

    session_id = str(uuid.uuid4())
    for i, question in enumerate(questions, 1):
        print(f"##################\n### Question {i} ###\n##################")

        q = question["question"]
        a = question["answer"]
        pdf = question["paper"]

        # Chat with the model
        response = runnable.invoke(
            {"input": q},
            config={"configurable": {"session_id": session_id}},
        )


        # print(
        #     f'~~~ Question ~~~\n{q}\n\n~~~ Output ~~~\n{output["answer"]}\n\n~~~ "Correct" Answer ~~~\n{a}\n\n~~~ Paper ~~~\n{pdf}\n'
        # )
        print(response["answer"])