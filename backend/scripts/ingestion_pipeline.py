"""
One-shot script that builds (or rebuilds) the ChromaDB knowledge base from scratch.

Run this when setting up the project for the first time, or whenever you want to
reload all sources from scratch. For incremental additions after setup, use the
admin upload endpoint instead (POST /api/admin/upload).

Usage:
    python3 backend/scripts/ingestion_pipeline.py
"""

import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from bs4 import SoupStrainer
from bs4 import BeautifulSoup
import requests
from langchain_core.documents import Document
import pdfplumber

# Some sites block requests without a recognizable User-Agent.
os.environ["USER_AGENT"] = "the-global-desk/1.0"
load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
CHROMA_DIR = BACKEND_DIR / "chroma_db"

def load_urls(filepath=DATA_DIR / "urls.txt"):
    """Read URLs from a plain text file, one per line. Lines starting with # are ignored."""
    urls = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    print(f"Loaded {len(urls)} URLs from {filepath}")
    return urls

def load_web_documents(urls):
    documents = []
    for url in urls:
        print(f"  [WEB] {url}")
        text = fetch_clean_text(url)
        if text:
            documents.append(Document(
                page_content=text,
                metadata={"source": url, "type": "web"}
            ))
    return documents

def load_pdf_documents(pdf_dir="../data/pdfs"):
    documents = []
    if not os.path.exists(pdf_dir):
        return documents

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            print(f"  [PDF] {filename}")
            with pdfplumber.open(filepath) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": filename, "type": "pdf"}
                ))
    return documents

def load_txt_documents(txt_dir="../data/txt"):
    documents = []
    if not os.path.exists(txt_dir):
        return documents

    for filename in os.listdir(txt_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(txt_dir, filename)
            print(f"  [TXT] {filename}")
            with open(filepath, "r") as f:
                text = f.read()
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": filename, "type": "txt"}
                ))
    return documents

def load_all_documents():
    print("Loading documents from all sources...\n")

    urls = load_urls()
    web_docs = load_web_documents(urls)
    pdf_docs = load_pdf_documents()
    txt_docs = load_txt_documents()

    documents = web_docs + pdf_docs + txt_docs

    if len(documents) == 0:
        raise ValueError("No documents were loaded.")

    print(f"\nTotal: {len(documents)} documents")
    print(f"  Web: {len(web_docs)}")
    print(f"  PDF: {len(pdf_docs)}")
    print(f"  TXT: {len(txt_docs)}\n")

    for doc in documents:
        print(f"[{doc.metadata.get('type')}] {doc.metadata.get('source')}")
        print(f"  Length: {len(doc.page_content)} chars\n")

    return documents

def fetch_clean_text(url):
    """
    Fetch a page and extract its main readable text, stripping navigation/chrome.

    We try a cascade of selectors so the function works across different CMS layouts:
    1. .wysiwyg — Davidson's CMS wraps body copy in this class
    2. <article> / <main> — semantic HTML fallbacks used by many modern sites
    3. <body> — last resort so we always return something rather than nothing

    The 1000-char threshold for the wysiwyg check filters out pages where that div
    exists but contains only a short intro blurb rather than the full content.
    """
    response = requests.get(url, headers={"User-Agent": "the-global-desk/1.0"}, timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")

    # Strip boilerplate regions before extracting text.
    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
        tag.decompose()

    content = soup.find("div", class_="wysiwyg")

    if not content or len(content.get_text(strip=True)) < 1000:
        content = (
            soup.find("article") or
            soup.find("main") or
            soup.find("body")
        )

    text = content.get_text(separator="\n", strip=True) if content else ""
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def load_documents(urls):
    """Legacy URL-only loader kept for reference. Use load_all_documents() instead."""
    print("Loading documents from URLs...")
    documents = []

    for url in urls:
        print(f"  Fetching: {url}")
        text = fetch_clean_text(url)
        if text:
            documents.append(Document(page_content=text, metadata={"source": url}))

    if len(documents) == 0:
        raise ValueError("No documents were loaded from URLs.")

    print(f"\nLoaded {len(documents)} documents\n")
    for doc in documents:
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Preview: {doc.page_content[:200]}...\n")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Break documents into overlapping chunks for embedding.

    chunk_size=1000 keeps each chunk under typical context-window limits while
    still holding enough context for the LLM to answer from.
    chunk_overlap=200 prevents a sentence from being cut off right at a boundary
    and losing its meaning across two adjacent chunks.
    """
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    ## Preview
    print(f"Created {len(chunks)} chunks\n")
    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} | Source: {chunk.metadata.get('source', 'unknown')} ---")
        print(f"{chunk.page_content[:300]}...\n")

    return chunks

def vectorize_db(chunks, persist_directory=CHROMA_DIR):

    print("Creating embeddings and storing in ChromaDB")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # cosine distance works better than the default L2 for text embeddings
    # because sentence-length differences don't inflate the distance unfairly.
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector db created and saved to {persist_directory}")
    
    return vector_db

def save_preview(documents, output_path=DATA_DIR / "preview.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for doc in documents:
            f.write(f"{'='*80}\n")
            f.write(f"SOURCE: {doc.metadata.get('source')}\n")
            f.write(f"LENGTH: {len(doc.page_content)} chars\n")
            f.write(f"{'='*80}\n")
            f.write(doc.page_content)
            f.write(f"\n\n")
    print(f"Preview saved to {output_path}")

def main():
    print("Main Function")

    urls = load_urls()

    # 1. Loading the files
    documents = load_all_documents()
    save_preview(documents)
    
    # 2. Chunking the files
    chunks = split_documents(documents)

    # 3. Embedding and storing in vector DB
    db = vectorize_db(chunks)

    print("\nDone! Database is ready for retrieval.")
    
if __name__ == "__main__":
    main()
