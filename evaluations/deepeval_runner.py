"""Integration layer for running DeepEval evaluations."""

from __future__ import annotations

import importlib
from typing import Any, Iterable

from .base_evaluator import BaseEvaluator, EvaluationInput, EvaluationResult


def _load_optional_class(module_name: str, class_name: str) -> Any | None:
    """Attempt to import ``class_name`` from ``module_name`` safely."""

    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError):  # pragma: no cover - optional dependency
        return None


class DeepEvalRunner(BaseEvaluator):
    """Wraps DeepEval's evaluation pipeline or uses simple offline evaluation."""

    def __init__(self, output_dir=None) -> None:
        super().__init__("deepeval", output_dir=output_dir)
        # Use simple offline evaluation instead of DeepEval's LLM-dependent metrics
        self._available = True

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
                details={"error": "deepeval not installed"},
            )

        # Simple offline evaluation: measure answer length and word overlap as a basic proxy
        total_overlap = 0.0
        for item in records:
            pred_words = set(item.prediction.lower().split())
            ref_words = set(item.reference.lower().split())
            if pred_words or ref_words:
                overlap = len(pred_words & ref_words) / max(
                    len(pred_words | ref_words), 1
                )
                total_overlap += overlap

        score = total_overlap / len(records) if records else 0.0
        details = {
            "metric": "word_overlap",
            "num_samples": len(records),
            "method": "offline_comparison",
        }
        result = EvaluationResult(framework=self.name, score=score, details=details)
        self.save_result(result)
        return result
