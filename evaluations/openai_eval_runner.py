from __future__ import annotations

from typing import Iterable, Any

from .base_evaluator import BaseEvaluator, EvaluationInput, EvaluationResult


class OpenAIEvalRunner(BaseEvaluator):
    """Hooks into OpenAI Evals when the package is available."""

    def __init__(self, output_dir=None) -> None:
        super().__init__("openai_evals", output_dir=output_dir)
        try:
            import evals  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            self._available = False
            self._evals: Any = None
        else:
            self._available = True
            self._evals = evals

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
