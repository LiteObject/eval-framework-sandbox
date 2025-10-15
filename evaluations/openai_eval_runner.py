"""Integration layer for wrapping OpenAI Evals command-line workflows."""

from __future__ import annotations

import importlib
from typing import Any, Iterable

from .base_evaluator import BaseEvaluator, EvaluationInput, EvaluationResult


def _load_optional_module(module_name: str) -> Any | None:
    """Return the imported module or ``None`` when unavailable."""

    try:
        return importlib.import_module(module_name)
    except ImportError:  # pragma: no cover - optional dependency
        return None


class OpenAIEvalRunner(BaseEvaluator):
    """Hooks into OpenAI Evals when the package is available."""

    def __init__(self, output_dir=None) -> None:
        super().__init__("openai_evals", output_dir=output_dir)
        self._evals: Any | None = _load_optional_module("evals")
        self._available = self._evals is not None

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
                details={"error": "openai-evals not installed"},
            )

        # Placeholder integration: record dataset for manual CLI usage
        eval_dataset = [record.__dict__ for record in records]
        details = {
            "message": "Dataset prepared; run `oaieval` CLI for full evaluation",
            "dataset_preview": eval_dataset,
        }
        result = EvaluationResult(framework=self.name, score=None, details=details)
        self.save_result(result)
        return result
