"""
TF-IDF (Term Frequency-Inverse Document Frequency) based
embedding index used for local document retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .document_loader import Document


@dataclass
class RetrievedContext:
    """Document result paired with its similarity score."""

    document: Document
    score: float


class EmbeddingIndex:
    """Simple TF-IDF (Term Frequency-Inverse Document Frequency) based retrieval index."""

    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("No documents supplied for indexing")

        # stop_words="english" -> Ignores common English words (the, a, is, etc.)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(
            doc.content for doc in self.documents
        )

    def query(self, text: str, top_k: int = 3) -> list[RetrievedContext]:
        """Return the top ``top_k`` contexts matching the provided text."""

        if not text.strip():
            return []
        query_vec = self.vectorizer.transform([text])

        # cosine_similarity is a math function that measures how similar two things are
        # by comparing them as vectors. It returns a score from 0 to 1:
        # 1.0 = Identical (perfect match), 0.5 = Somewhat similar, 0.0 = Completely different
        similarity_scores = cosine_similarity(query_vec, self.matrix)[0]
        rankings = np.argsort(similarity_scores)[::-1][:top_k]
        return [
            RetrievedContext(
                document=self.documents[idx], score=float(similarity_scores[idx])
            )
            for idx in rankings
            if similarity_scores[idx] > 0
        ]
