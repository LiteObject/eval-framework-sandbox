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
    """Wraps DeepEval's evaluation pipeline when available."""

    def __init__(self, output_dir=None) -> None:
        super().__init__("deepeval", output_dir=output_dir)
        dataset_cls = _load_optional_class("deepeval.dataset", "Dataset")
        metric_cls = _load_optional_class(
            "deepeval.metrics",
            "AnswerCorrectnessMetric",
        )
        evaluator_cls = _load_optional_class("deepeval.evaluator", "Evaluator")

        if not all((dataset_cls, metric_cls, evaluator_cls)):
            self._available = False
            self._dataset_cls: Any | None = None
            self._metric_cls: Any | None = None
            self._evaluator_cls: Any | None = None
        else:
            self._available = True
            self._dataset_cls = dataset_cls
            self._metric_cls = metric_cls
            self._evaluator_cls = evaluator_cls

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

        dataset_cls = self._dataset_cls
        metric_cls = self._metric_cls
        evaluator_cls = self._evaluator_cls
        assert dataset_cls and metric_cls and evaluator_cls  # narrow for type-checkers

        ds = dataset_cls(
            samples=[
                {
                    "input": item.question,
                    "actual_output": item.prediction,
                    "expected_output": item.reference,
                }
                for item in records
            ]
        )
        evaluator = evaluator_cls(metrics=[metric_cls()])
        report = evaluator.evaluate(ds)
        score = report.overall_score
        details = {
            "metric_breakdown": report.metric_scores,
            "num_samples": len(records),
        }
        result = EvaluationResult(framework=self.name, score=score, details=details)
        self.save_result(result)
        return result
