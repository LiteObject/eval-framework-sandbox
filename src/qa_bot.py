from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .document_loader import DocumentLoader, Document
from .embeddings import EmbeddingIndex, RetrievedContext


@dataclass
class Answer:
    question: str
    response: str
    context: list[RetrievedContext]


class QABot:
    """Minimal retrieval-augmented QA bot for local documentation."""

    def __init__(
        self, documents_path: Path | None = None, top_k: int | None = None
    ) -> None:
        docs_path = documents_path or settings.documents_path
        if not docs_path.exists():
            raise FileNotFoundError(f"Documentation directory not found: {docs_path}")

        loader = DocumentLoader(docs_path)
        documents = loader.load()
        if not documents:
            raise ValueError(f"No Markdown documents found in {docs_path}")

        self.index = EmbeddingIndex(documents)
        self.top_k = top_k or settings.top_k

    def retrieve(self, question: str) -> list[RetrievedContext]:
        return self.index.query(question, self.top_k)

    def answer(self, question: str) -> Answer:
        contexts = self.retrieve(question)
        if not contexts:
            return Answer(
                question=question,
                response="I couldn't find relevant documentation.",
                context=[],
            )

        best = contexts[0]
        snippet = self._extract_snippet(best.document, question)
        response = (
            f"According to {best.document.title}, {snippet}"
            if snippet
            else best.document.content
        )
        return Answer(question=question, response=response, context=contexts)

    @staticmethod
    def _extract_snippet(document: Document, question: str) -> str:
        lowered_question = question.lower()
        lines = [line.strip() for line in document.content.splitlines() if line.strip()]
        question_terms = lowered_question.split()

        install_candidate: str | None = None
        token_candidate: str | None = None

        # Prefer lines that contain installation hints or share tokens
        for line in lines:
            lower_line = line.lower()
            if lower_line.startswith("#"):
                continue
            if "pip install" in lower_line:
                return line
            if install_candidate is None and "install" in lower_line:
                install_candidate = line
            tokens = set(lower_line.split())
            if token_candidate is None and (
                any(token in lowered_question for token in tokens)
                or any(term in lower_line for term in question_terms)
            ):
                token_candidate = line

        if install_candidate is not None:
            return install_candidate
        if token_candidate is not None:
            return token_candidate

        # Fallback to the first sentence/paragraph
        for line in lines:
            if not line.strip().startswith("#"):
                return line
        return lines[0] if lines else ""
