"""Shared data structures and abstract base for evaluation runners."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class EvaluationInput:
    """Single question and answer pair used as evaluation input."""

    question: str
    prediction: str
    reference: str


@dataclass
class EvaluationResult:
    """Aggregated evaluation output from a framework."""

    framework: str
    score: float | None
    details: dict[str, object]


class BaseEvaluator(ABC):
    """Shared contract for invoking external evaluation frameworks."""

    name: str

    def __init__(self, name: str, output_dir: Path | None = None) -> None:
        self.name = name
        self.output_dir = output_dir or Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def evaluate(self, dataset: Iterable[EvaluationInput]) -> EvaluationResult:
        """Run evaluation for a dataset returning an aggregated score."""

    def save_result(self, result: EvaluationResult) -> Path:
        """Persist the evaluation result to disk as JSON and return the path."""

        path = self.output_dir / f"{self.name}_result.json"

        with path.open("w", encoding="utf-8") as fp:
            json.dump(
                {
                    "framework": result.framework,
                    "score": result.score,
                    "details": result.details,
                },
                fp,
                indent=2,
            )
        return path
