"""LangChain evaluation runner with configurable chat backends."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Iterable, Optional, cast

from dotenv import load_dotenv

from src.config import settings

from .base_evaluator import BaseEvaluator, EvaluationInput, EvaluationResult

load_dotenv()


def _load_optional_class(module_name: str, class_name: str) -> Any | None:
    """Import a class dynamically, returning ``None`` when unavailable."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError):  # pragma: no cover - optional dependency
        return None


class LangChainEvalRunner(BaseEvaluator):
    """Uses LangChain's built-in evaluators when installed."""

    def __init__(self, output_dir=None) -> None:
        super().__init__("langchain", output_dir=output_dir)
        self._llm_builder: Optional[Callable[[], Any]] = None
        self._llm_provider: Optional[str] = None
        self._llm_error: Optional[str] = None
        self._qa_eval_chain_cls: Any | None = _load_optional_class(
            "langchain.evaluation.qa",
            "QAEvalChain",
        )

        if self._qa_eval_chain_cls is None:
            self._available = False
            if self._llm_error is None:
                self._llm_error = (
                    "LangChain QA evaluator is unavailable; install langchain."
                )
            return

        self._available = True

        if settings.langchain_use_ollama:
            self._configure_ollama_backend()

        if self._llm_builder is None:
            self._configure_openai_backend()

        # Log model configuration
        if self._llm_provider == "ollama":
            model = settings.ollama_model or "unknown"
            base_url = settings.ollama_base_url or "http://localhost:11434"
            print(
                f"[LangChain] Using Ollama backend: model={model}, base_url={base_url}"
            )
        elif self._llm_provider == "openai":
            model = settings.langchain_openai_model or "unknown"
            print(f"[LangChain] Using OpenAI backend: model={model}")
        else:
            print("[LangChain] Backend not configured")

    def _configure_ollama_backend(self) -> None:
        chat_ollama_cls = _load_optional_class(
            "langchain_community.chat_models",
            "ChatOllama",
        )

        if chat_ollama_cls is None:
            self._llm_error = (
                "LANGCHAIN_USE_OLLAMA=true but ChatOllama is unavailable. "
                "Install langchain-community or disable the flag."
            )
            return

        def build_ollama() -> Any:
            kwargs: dict[str, Any] = {"model": settings.ollama_model}
            if settings.ollama_base_url:
                kwargs["base_url"] = settings.ollama_base_url
            return chat_ollama_cls(**kwargs)

        self._llm_builder = build_ollama
        self._llm_provider = "ollama"

    def _configure_openai_backend(self) -> None:
        chat_openai_cls = _load_optional_class(
            "langchain.chat_models",
            "ChatOpenAI",
        )

        if chat_openai_cls is None:
            chat_openai_cls = _load_optional_class(
                "langchain_openai",
                "ChatOpenAI",
            )

        if chat_openai_cls is None:
            if self._llm_error is None:
                self._llm_error = (
                    "Could not import ChatOpenAI; install langchain or set "
                    "LANGCHAIN_USE_OLLAMA=true."
                )
            return

        self._configure_openai(chat_openai_cls)

    def _configure_openai(self, chat_openai_cls: Any) -> None:
        if not settings.openai_api_key:
            self._llm_error = (
                "ChatOpenAI requires OPENAI_API_KEY; set LANGCHAIN_USE_OLLAMA=true "
                "to use a local model."
            )
            return

        def build_openai() -> Any:
            kwargs: dict[str, Any] = {
                "temperature": 0,
                "openai_api_key": settings.openai_api_key,
            }
            for model_key in ("model", "model_name"):
                try:
                    return chat_openai_cls(
                        **kwargs,
                        **{model_key: settings.langchain_openai_model},
                    )
                except TypeError:
                    continue
            return chat_openai_cls(**kwargs)

        self._llm_builder = build_openai
        self._llm_provider = "openai"

    def evaluate(self, dataset: Iterable[EvaluationInput]) -> EvaluationResult:
        records = list(dataset)
        if not records:
            return EvaluationResult(
                framework=self.name,
                score=None,
                details={"error": "empty dataset"},
            )

        if not self._available:
            return EvaluationResult(
                self.name,
                None,
                {"error": "langchain not installed"},
            )

        if self._qa_eval_chain_cls is None:
            return EvaluationResult(
                framework=self.name,
                score=None,
                details={"error": self._llm_error or "Evaluator unavailable"},
            )

        if not self._llm_builder:
            return EvaluationResult(
                framework=self.name,
                score=None,
                details={
                    "error": self._llm_error
                    or "No LangChain chat model available for evaluation",
                    "provider": self._llm_provider,
                },
            )

        llm = self._llm_builder()
        qa_chain = self._qa_eval_chain_cls.from_llm(llm)
        eval_results = [
            qa_chain.evaluate_strings(
                prediction=item.prediction,
                reference=item.reference,
                input=item.question,
            )
            for item in records
        ]
        score = sum(1 for res in eval_results if res.get("score", 0) >= 0.5) / len(
            records
        )
        details = {
            "raw": cast(object, eval_results),
            "provider": self._llm_provider,
        }
        result = EvaluationResult(framework=self.name, score=score, details=details)
        self.save_result(result)
        return result
