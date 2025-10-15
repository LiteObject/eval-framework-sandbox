"""Utilities for loading Markdown documentation into structured objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class Document:
    """Lightweight representation of a Markdown document."""

    doc_id: str
    title: str
    content: str


class DocumentLoader:
    """Utility for loading Markdown documentation."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self) -> list[Document]:
        """Return all Markdown files under ``root`` as ``Document`` instances."""

        documents: list[Document] = []
        for path in sorted(self.root.glob("*.md")):
            documents.append(
                Document(
                    doc_id=path.stem,
                    title=path.stem.replace("_", " ").title(),
                    content=path.read_text(encoding="utf-8"),
                )
            )
        return documents

    def load_iter(self) -> Iterable[Document]:
        """Iterate lazily over documents without materializing the list."""

        yield from self.load()
