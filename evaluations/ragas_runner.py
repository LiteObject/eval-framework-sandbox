from __future__ import annotations

from typing import Iterable, Any

from .base_evaluator import BaseEvaluator, EvaluationInput, EvaluationResult


class RagasRunner(BaseEvaluator):
    """Integrates the RAGAS evaluation pipeline when installed."""

    def __init__(self, output_dir=None) -> None:
        super().__init__("ragas", output_dir=output_dir)
        try:
            from ragas import evaluate  # type: ignore
            from ragas.metrics import context_precision  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            self._available = False
            self._evaluate: Any = None
            self._context_precision: Any = None
        else:
            self._available = True
            self._evaluate = evaluate
            self._context_precision = context_precision

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
