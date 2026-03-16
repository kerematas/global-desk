from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
persistent_directory = BACKEND_DIR / "chroma_db"

# Load embeddings and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "Can F-1 students work off campus, and when do they need OPT or CPT authorization?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    source = doc.metadata.get("source", "unknown")
    print(f"Document {i} | Source: {source}\n{doc.page_content}\n")


# Synthetic Questions: 

# 1. "What are the requirements for maintaining F-1 student status at Davidson?"
# 2. "Can an international student work off campus during the semester?"
# 3. "What is the difference between CPT and OPT for F-1 students?"
# 4. "When should a student apply for post-completion OPT?"
# 5. "Who needs to file U.S. taxes or social security paperwork as an international student?"
# 6. "Do F-1 students have to pay Social Security and Medicare taxes?"
# 7. "What documents should an F-1 student carry when traveling outside the United States?"
# 8. "What support does Davidson provide for new international students after arrival?"
# 9. "What does the Davidson host family program offer international students?"
# 10. "Where can students find official government guidance about SEVIS and practical training?"
