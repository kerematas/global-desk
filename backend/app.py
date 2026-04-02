"""
Small FastAPI app that serves both the frontend and the chat API.

This keeps development simple:
one command starts the site and the RAG backend together.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.rag_service import CHROMA_SQLITE_FILE, ENV_PATH, RAGService, RAGServiceError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"

app = FastAPI(title="Global Desk")
rag_service = RAGService()


class ChatHistoryItem(BaseModel):
    """One prior chat turn sent from the browser."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """Payload for the simple single-chat API."""

    message: str = Field(min_length=1)
    history: list[ChatHistoryItem] = Field(default_factory=list)


@app.get("/api/health")
def health_check() -> dict[str, bool]:
    """
    Very small health endpoint for development checks.

    It reports whether the key local dependencies are configured without trying
    to run a full model call.
    """

    return {
        "ok": True,
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "knowledge_base_ready": CHROMA_SQLITE_FILE.exists(),
        "env_file_found": ENV_PATH.exists(),
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    Receive one user message plus prior history and return the RAG answer.

    The backend stays stateless: the browser sends the history each time.
    """

    try:
        return rag_service.answer_question(
            message=request.message,
            history=[item.model_dump() for item in request.history],
        )
    except RAGServiceError as error:
        return JSONResponse(status_code=400, content={"error": str(error)})
    except Exception as error:
        print(f"Unexpected chat error: {error}")
        return JSONResponse(
            status_code=500,
            content={
                "error": (
                    "Something went wrong while generating a response. "
                    "Please try again."
                )
            },
        )


@app.get("/", include_in_schema=False)
def read_index() -> FileResponse:
    """Serve the frontend entry page."""

    return FileResponse(INDEX_FILE)


# Serve the frontend files under /static so the HTML can load its JS, CSS, and logo.
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
