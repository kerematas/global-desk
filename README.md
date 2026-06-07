# Global Desk

RAG-based assistant helping F-1 international university communities with visa, immigration, tax, and policy questions.

## Group

Davidson Indoors

## Team

- Kerem Atas, Product Manager
- Hakan Bora Yavuzkara, Scrum Master
- Elie Jerjees, Developer
- Tanaka Makoni, Developer

## About

Global Desk is an AI-powered Q&A tool built for international students offices. It uses Retrieval-Augmented Generation (RAG) to answer questions based on official documents — covering topics like visa status, employment authorization, tax obligations, and institutional policies.

Built as a project for CSC 312: Software Design at Davidson College.

### Core Features

- Document-grounded AI chat for F-1 visa, CPT/OPT, travel, tax, and international student policy questions.
- Persistent ChromaDB knowledge base built from official web pages, PDFs, and text files.
- Retrieval-Augmented Generation flow using `text-embedding-3-small` embeddings and `gpt-4o` answer generation.
- Follow-up question handling by rewriting conversational messages into standalone retrieval queries.
- Source-aware responses that return deduplicated document sources to the frontend.
- Admin dashboard protected with HTTP Basic Auth for uploading, listing, and deleting knowledge base documents.
- Incremental PDF/DOCX ingestion that extracts text, chunks documents, embeds chunks, and appends them to ChromaDB.
- Browser chat UI with local conversation persistence, prompt chips, source rendering, typing state, and stateless API history submission.
- Batch evaluation harness for running a 100-question test set through the retrieval pipeline.

## Architecture

```text
frontend/index.html + frontend/app.js
        |
        | POST /api/chat
        v
backend/app.py  ----------------------->  backend/rag_service.py
FastAPI routes                           RAG orchestration
        |                                      |
        |                                      v
        |                              ChromaDB vector store
        |                                      |
        v                                      v
admin/index.html                    OpenAI embeddings + chat model
Admin upload/list/delete
```

### Backend

- `backend/app.py` defines the FastAPI server, serves the frontend, exposes chat and health endpoints, and protects admin routes with constant-time credential comparison.
- `backend/rag_service.py` centralizes the RAG workflow so the API can reuse one retrieval and answer-generation implementation.
- `backend/scripts/ingestion_pipeline.py` rebuilds the vector database from web URLs, PDFs, and text files.
- `backend/scripts/retrieval_pipeline.py` provides an interactive terminal chat path for local testing and evaluation.

### RAG Pipeline

1. Load official source material from configured web, PDF, and text sources.
2. Scrape readable page content with BeautifulSoup, including Davidson CMS-specific fallbacks.
3. Extract PDF text with `pdfplumber`.
4. Split documents into overlapping 1,000-character chunks with 200-character overlap.
5. Embed chunks with OpenAI `text-embedding-3-small`.
6. Persist vectors in ChromaDB using cosine similarity.
7. Retrieve the top 3 relevant chunks for each user question.
8. Generate an answer with `gpt-4o`, constrained to the retrieved documents.
9. Suppress sources when the model cannot answer from the available documents.

### Frontend

- Uses vanilla JavaScript to keep the prototype lightweight and easy to deploy.
- Stores the current single-chat conversation in `localStorage`.

### Admin Workflow

- Admin users authenticate through HTTP Basic Auth.
- PDF and DOCX uploads are text-extracted server-side.

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Serves the main chat UI. |
| `GET` | `/api/health` | Reports API key and knowledge base readiness. |
| `POST` | `/api/chat` | Accepts a message plus chat history and returns a RAG answer with sources. |
| `GET` | `/api/admin/verify` | Validates admin credentials. |
| `POST` | `/api/admin/upload` | Uploads and ingests a PDF or DOCX into the vector store. |
| `GET` | `/api/admin/documents` | Lists unique document sources in the knowledge base. |
| `DELETE` | `/api/admin/document` | Removes a document source from ChromaDB and disk when applicable. |

## Tech Stack

- **Backend:** Python, FastAPI, Pydantic, Uvicorn
- **AI/RAG:** LangChain, OpenAI, ChromaDB, `text-embedding-3-small`, `gpt-4o`
- **Document Processing:** BeautifulSoup, Requests, pdfplumber, python-docx
- **Frontend:** HTML, CSS, vanilla JavaScript, localStorage
- **Evaluation:** Python subprocess-based batch runner with resumable output

## Run Locally

Start the connected frontend and backend together with:

```bash
uvicorn backend.app:app --reload
```

Then open `http://127.0.0.1:8000`.

If the knowledge base is missing, build it first with:

```bash
python3 backend/scripts/ingestion_pipeline.py
```

Configure the required environment variables before running the app:

```bash
OPENAI_API_KEY=your_api_key_here
ADMIN_USERNAME=admin
ADMIN_PASSWORD=changeme
```

## License

MIT
