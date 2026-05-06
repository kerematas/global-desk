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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
from langchain_core.documents import Document
import pdfplumber

# Some sites block requests without a recognizable User-Agent.
os.environ["USER_AGENT"] = "the-global-desk/1.0"
load_dotenv()

# paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
DATA_DIR = BACKEND_DIR / "data"         # where pdfs, txts, and urls.txt live
CHROMA_DIR = BACKEND_DIR / "chroma_db"  # where the vector db gets saved

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
    """Scrape each URL and turn it into a LangChain Document."""
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

def load_pdf_documents(pdf_dir=DATA_DIR / "pdfs"):
    """Read all PDFs from the pdfs folder and extract their text."""
    documents = []
    if not os.path.exists(pdf_dir):
        return documents

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            print(f"  [PDF] {filename}")
            # extract text from every page and join them together
            with pdfplumber.open(filepath) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": filename, "type": "pdf"}
                ))
    return documents

def load_txt_documents(txt_dir=DATA_DIR / "txt"):
    """Read all .txt files from the txt folder."""
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
    """Load documents from every source type (web, pdf, txt) and combine them."""
    print("Loading documents from all sources...\n")

    # gather documents from all three source types
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

    # remove nav, header, footer, etc. so we only get the actual content
    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
        tag.decompose()

    # try Davidson's CMS wrapper first, then fall back to standard HTML tags
    content = soup.find("div", class_="wysiwyg")

    if not content or len(content.get_text(strip=True)) < 1000:
        content = (
            soup.find("article") or
            soup.find("main") or
            soup.find("body")
        )

    # pull out clean text and remove empty lines
    text = content.get_text(separator="\n", strip=True) if content else ""
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

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

def main():
    # full pipeline: load everything -> chunk it -> embed and store in vector db
    documents = load_all_documents()
    chunks = split_documents(documents)
    db = vectorize_db(chunks)

    print("\nDone! Database is ready for retrieval.")
    
if __name__ == "__main__":
    main()
