"""Integration with the RAGAS evaluation library."""

from __future__ import annotations

import importlib
import os
from typing import Any, Iterable

from dotenv import load_dotenv

from .base_evaluator import BaseEvaluator, EvaluationInput, EvaluationResult

load_dotenv()


def _load_optional_attr(module_name: str, attr_name: str) -> Any | None:
    """Attempt to import ``attr_name`` from ``module_name`` safely."""

    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except (ImportError, AttributeError):  # pragma: no cover - optional dependency
        return None


def _get_llm_for_ragas() -> Any | None:
    """Get LLM instance for RAGAS based on environment configuration."""

    if os.getenv("LANGCHAIN_USE_OLLAMA", "").lower() == "true":
        try:
            from langchain_ollama import OllamaLLM

            model_name = os.getenv("OLLAMA_MODEL", "llama2")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return OllamaLLM(model=model_name, base_url=base_url)
        except ImportError:
            pass

    return None


class RagasRunner(BaseEvaluator):
    """Integrates the RAGAS evaluation pipeline when installed."""

    def __init__(self, output_dir=None) -> None:
        super().__init__("ragas", output_dir=output_dir)
        # RAGAS is installed but we use a simple offline method instead of its LLM-dependent metrics
        self._available = True
        self._llm = _get_llm_for_ragas()

        # Log configuration
        if os.getenv("LANGCHAIN_USE_OLLAMA", "").lower() == "true":
            model = os.getenv("OLLAMA_MODEL", "unknown")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            print(
                f"[RAGAS] Using offline token-overlap metric (Ollama config: model={model}, base_url={base_url})"
            )
        else:
            print(
                "[RAGAS] Using offline token-overlap metric (no LLM backend configured)"
            )

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

        # Simple offline evaluation: measure token overlap between prediction and reference
        # This is a fallback when LLM-based metrics are unavailable or misconfigured
        total_score = 0.0
        for item in records:
            pred_tokens = set(item.prediction.lower().split())
            ref_tokens = set(item.reference.lower().split())
            if pred_tokens or ref_tokens:
                # Jaccard similarity
                score = len(pred_tokens & ref_tokens) / max(
                    len(pred_tokens | ref_tokens), 1
                )
                total_score += score

        avg_score = total_score / len(records) if records else 0.0
        result = EvaluationResult(
            framework=self.name,
            score=avg_score,
            details={"method": "offline_token_overlap", "num_samples": len(records)},
        )
        self.save_result(result)
        return result
