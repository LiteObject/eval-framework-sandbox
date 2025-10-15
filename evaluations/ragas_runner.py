"""Integration with the RAGAS evaluation library."""

from __future__ import annotations

import importlib
from typing import Any, Iterable

from .base_evaluator import BaseEvaluator, EvaluationInput, EvaluationResult


def _load_optional_attr(module_name: str, attr_name: str) -> Any | None:
    """Attempt to import ``attr_name`` from ``module_name`` safely."""

    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except (ImportError, AttributeError):  # pragma: no cover - optional dependency
        return None


class RagasRunner(BaseEvaluator):
    """Integrates the RAGAS evaluation pipeline when installed."""

    def __init__(self, output_dir=None) -> None:
        super().__init__("ragas", output_dir=output_dir)
        evaluate_fn = _load_optional_attr("ragas", "evaluate")
        context_precision_metric = _load_optional_attr(
            "ragas.metrics", "context_precision"
        )

        if evaluate_fn and context_precision_metric:
            self._available = True
            self._evaluate = evaluate_fn
            self._context_precision = context_precision_metric
        else:
            self._available = False
            self._evaluate = None
            self._context_precision = None

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
                details={"error": "ragas not installed"},
            )

        evaluate_fn = self._evaluate
        context_precision_metric = self._context_precision
        assert evaluate_fn and context_precision_metric

        ragas_dataset = {
            "question": [item.question for item in records],
            "contexts": [[item.reference] for item in records],
            "answer": [item.prediction for item in records],
            "ground_truth": [item.reference for item in records],
        }

        report = evaluate_fn(ragas_dataset, metrics=[context_precision_metric])
        score = float(report[0].score) if report else 0.0
        result = EvaluationResult(
            framework=self.name, score=score, details={"raw": report}
        )
        self.save_result(result)
        return result
