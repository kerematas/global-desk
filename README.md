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

## License

MIT
