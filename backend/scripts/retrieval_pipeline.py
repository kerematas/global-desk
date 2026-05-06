"""
Interactive terminal chat against the ChromaDB knowledge base.

This script is a standalone CLI tool — run it directly to ask questions and get
RAG-grounded answers in the terminal. It is also used by the test/evaluation pipeline,
which drives it via subprocess stdin.

Note: the embeddings, db, and model are all initialised at module import time.
That means the OpenAI API key must already be in the environment when the script
starts. There is no lazy initialisation here.
"""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
persistent_directory = BACKEND_DIR / "chroma_db"

# set up the embedding model, vector db, and chat model once at startup
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=str(persistent_directory), embedding_function=embeddings)
model = ChatOpenAI(model="gpt-4o")

# keeps track of the conversation so follow-up questions make sense
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # if there's prior conversation, rewrite the question so it stands on its own
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # search the vector db for the 3 most relevant chunks
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    # show a quick preview of what we found
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")

    # build the prompt with the retrieved docs as context
    document_context = "\n".join([f"- {doc.page_content}" for doc in docs])
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {document_context}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    # send everything to the LLM (system prompt + history + new question with docs)
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    # save this exchange so future questions have context
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()
