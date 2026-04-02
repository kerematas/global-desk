"""
Shared RAG helper for both the API server and the terminal chat script.

The goal here is to keep the retrieval and answer-generation logic in one place
so the frontend API and the local CLI do not drift apart over time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
CHROMA_DIR = BACKEND_DIR / "chroma_db"
CHROMA_SQLITE_FILE = CHROMA_DIR / "chroma.sqlite3"
ENV_PATH = PROJECT_ROOT / ".env"

# Load the project .env explicitly so the app works no matter where it is run from.
load_dotenv(ENV_PATH)


class RAGServiceError(RuntimeError):
    """Friendly runtime errors that we can safely show in the UI."""


class RAGService:
    """
    Very small wrapper around the existing RAG flow.

    This class is intentionally lightweight:
    - it lazily opens the embedding model and vector DB when needed
    - it stays stateless between requests
    - it accepts browser-provided chat history for follow-up questions
    """

    def __init__(
        self,
        persist_directory: Path = CHROMA_DIR,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o",
        num_results: int = 3,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.num_results = num_results

    def answer_question(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Run the full RAG flow and return plain JSON-ready data.

        The API layer uses this method directly, and the terminal script can do
        the same thing without duplicating any of the business logic.
        """

        user_message = message.strip()
        if not user_message:
            raise RAGServiceError("Message cannot be empty.")

        normalized_history = history or []

        self._require_openai_api_key()
        self._require_chroma_db()

        model = self._create_chat_model()
        db = self._create_vector_db()

        search_question = self._build_search_question(
            user_message=user_message,
            history=normalized_history,
            model=model,
        )

        docs = self._retrieve_documents(search_question, db)
        answer = self._generate_answer(
            user_message=user_message,
            history=normalized_history,
            docs=docs,
            model=model,
        )

        return {
            "answer": answer,
            "sources": self._build_source_list(docs),
        }

    def _require_openai_api_key(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RAGServiceError(
                "OPENAI_API_KEY is missing. Add it to the project .env file "
                "before starting the app."
            )

    def _require_chroma_db(self) -> None:
        chroma_sqlite_file = self.persist_directory / "chroma.sqlite3"

        if not chroma_sqlite_file.exists():
            raise RAGServiceError(
                "The knowledge base is missing. Run "
                "'python3 backend/scripts/ingestion_pipeline.py' first."
            )

    def _create_chat_model(self) -> ChatOpenAI:
        return ChatOpenAI(model=self.chat_model)

    def _create_vector_db(self) -> Chroma:
        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        return Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=embeddings,
        )

    def _build_search_question(
        self,
        user_message: str,
        history: list[dict[str, str]],
        model: ChatOpenAI,
    ) -> str:
        """
        Turn follow-up questions into standalone search queries.

        If there is no history, we can search with the user's message directly.
        """

        history_messages = self._history_to_langchain_messages(history)
        if not history_messages:
            return user_message

        rewrite_messages = [
            SystemMessage(
                content=(
                    "Given the chat history, rewrite the new question to be a "
                    "standalone search query. Only return the rewritten "
                    "question."
                )
            ),
            *history_messages,
            HumanMessage(content=f"New question: {user_message}"),
        ]

        result = model.invoke(rewrite_messages)
        return result.content.strip()

    def _retrieve_documents(self, search_question: str, db: Chroma) -> list[Any]:
        retriever = db.as_retriever(search_kwargs={"k": self.num_results})
        return retriever.invoke(search_question)

    def _generate_answer(
        self,
        user_message: str,
        history: list[dict[str, str]],
        docs: list[Any],
        model: ChatOpenAI,
    ) -> str:
        """
        Ask the LLM to answer using only the retrieved documents.

        The history is included so the assistant can handle follow-up phrasing,
        but the prompt still tells the model to ground the answer in the docs.
        """

        history_messages = self._history_to_langchain_messages(history)
        document_context = self._format_documents_for_prompt(docs)

        final_prompt = (
            "Based on the following retrieved documents, answer the user's "
            f"question.\n\nQuestion:\n{user_message}\n\n"
            f"Documents:\n{document_context}\n\n"
            "Use only the retrieved documents for factual claims. If the "
            "documents do not contain enough information, say so clearly."
        )

        answer_messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant for international students. "
                    "Answer clearly and only use the retrieved documents for "
                    "factual information. Keep the answer easy to scan in a "
                    "simple chat UI. Prefer short paragraphs and bullet lists "
                    "when helpful. Do not use markdown bold, code fences, or "
                    "citation markers."
                )
            ),
            *history_messages,
            HumanMessage(content=final_prompt),
        ]

        result = model.invoke(answer_messages)
        return self._clean_answer_text(result.content)

    def _history_to_langchain_messages(
        self,
        history: list[dict[str, str]],
    ) -> list[HumanMessage | AIMessage]:
        """
        Convert the simple API history shape into LangChain message objects.

        Unknown roles and blank messages are skipped to keep the service
        resilient to minor frontend issues.
        """

        converted_messages: list[HumanMessage | AIMessage] = []

        for item in history:
            role = item.get("role", "").strip()
            content = item.get("content", "").strip()

            if not content:
                continue

            if role == "user":
                converted_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                converted_messages.append(AIMessage(content=content))

        return converted_messages

    def _format_documents_for_prompt(self, docs: list[Any]) -> str:
        """
        Build a readable block of retrieved text for the final answer prompt.

        Keeping the formatting separate also avoids the Python 3.11 f-string
        issue that the old retrieval script ran into.
        """

        if not docs:
            return "No documents were retrieved."

        sections: list[str] = []
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown source")
            sections.append(
                f"[Document {index}]\n"
                f"Source: {source}\n"
                f"Content:\n{doc.page_content}"
            )

        return "\n\n".join(sections)

    def _build_source_list(self, docs: list[Any]) -> list[dict[str, str]]:
        """Return a small deduplicated list of source URLs for future UI use."""

        sources: list[dict[str, str]] = []
        seen: set[str] = set()

        for doc in docs:
            source = str(doc.metadata.get("source", "")).strip()
            if source and source not in seen:
                seen.add(source)
                sources.append({"source": source})

        return sources

    def _clean_answer_text(self, text: str) -> str:
        """
        Clean up common markdown markers so answers display well in the
        intentionally simple frontend.
        """

        cleaned = text.strip()
        cleaned = cleaned.replace("**", "")
        cleaned = cleaned.replace("__", "")
        cleaned = cleaned.replace("`", "")
        cleaned = re.sub(r"\s*\(References?:[^)]*\)", "", cleaned)
        cleaned = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", cleaned)
        cleaned = re.sub(r"(?m)^#{1,6}\s*", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned
