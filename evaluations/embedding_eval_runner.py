"""Embedding-based evaluation using semantic similarity."""

from __future__ import annotations

import importlib
from typing import Any, Iterable

from dotenv import load_dotenv

from .base_evaluator import BaseEvaluator, EvaluationInput, EvaluationResult

load_dotenv()


def _load_optional_class(module_name: str, class_name: str) -> Any | None:
    """Attempt to import ``class_name`` from ``module_name`` safely."""

    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError):  # pragma: no cover - optional dependency
        return None


class EmbeddingEvalRunner(BaseEvaluator):
    """Uses embedding-based similarity to evaluate answers.

    This is a middle-ground approach between LLM-as-a-Judge and rule-based metrics.
    It converts both prediction and reference answers into dense vector embeddings
    and calculates cosine similarity. This captures semantic meaning better than
    word overlap but is faster and cheaper than LLM-based evaluation.
    """

    def __init__(self, output_dir=None) -> None:
        super().__init__("embedding", output_dir=output_dir)

        # Try to load sentence-transformers for embedding generation
        SentenceTransformer = _load_optional_class(
            "sentence_transformers",
            "SentenceTransformer",
        )

        if SentenceTransformer is None:
            self._available = False
            self._model = None
            self._error = (
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            print(f"[Embedding] ⚠ {self._error}")
            return

        self._available = True
        try:
            # Use a lightweight, fast embedding model
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            print(
                "[Embedding] Using sentence-transformers (all-MiniLM-L6-v2) for semantic similarity"
            )
        except (OSError, IOError, RuntimeError) as exc:
            self._available = False
            self._model = None
            self._error = f"Failed to load embedding model: {exc}"
            print(f"[Embedding] ⚠ {self._error}")

    def evaluate(self, dataset: Iterable[EvaluationInput]) -> EvaluationResult:
        records = list(dataset)
        if not records:
            return EvaluationResult(
                framework=self.name, score=None, details={"error": "empty dataset"}
            )

        if not self._available:
            return EvaluationResult(
                framework=self.name,
                score=None,
                details={"error": self._error or "embedding evaluator not available"},
            )

        if self._model is None:
            return EvaluationResult(
                framework=self.name,
                score=None,
                details={"error": "embedding model failed to load"},
            )

        # Calculate cosine similarity between prediction and reference embeddings
        total_similarity = 0.0
        for item in records:
            try:
                # Generate embeddings for prediction and reference
                pred_embedding = self._model.encode(
                    item.prediction, convert_to_tensor=False
                )
                ref_embedding = self._model.encode(
                    item.reference, convert_to_tensor=False
                )

                # Calculate cosine similarity
                similarity = self._cosine_similarity(pred_embedding, ref_embedding)
                total_similarity += similarity
            except (OSError, ValueError, RuntimeError) as exc:
                # If embedding fails for any item, skip it
                print(f"[Embedding] Warning: failed to embed item: {exc}")
                continue

        avg_score = total_similarity / len(records) if records else 0.0

        # Clamp to [0, 1] range in case of numerical issues
        avg_score = max(0.0, min(1.0, avg_score))

        details = {
            "metric": "cosine_similarity",
            "num_samples": len(records),
            "method": "semantic_embedding",
            "embedding_model": "all-MiniLM-L6-v2",
        }
        result = EvaluationResult(framework=self.name, score=avg_score, details=details)
        self.save_result(result)
        return result

    @staticmethod
    def _cosine_similarity(vec1: Any, vec2: Any) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np

        # Convert to numpy arrays if needed
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
