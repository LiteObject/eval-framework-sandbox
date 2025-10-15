"""Utility helpers for preparing datasets used by evaluation runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from .base_evaluator import EvaluationInput


def load_dataset_from_files(
    questions_path: Path,
    ground_truth_path: Path,
    predictions: Mapping[str, str] | None = None,
) -> Iterable[EvaluationInput]:
    """Yield ``EvaluationInput`` rows built from question, truth, and prediction files."""

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))

    for entry in questions:
        question_id = entry["id"]
        yield EvaluationInput(
            question=entry["question"],
            prediction=(predictions or {}).get(question_id, ""),
            reference=ground_truth.get(question_id, ""),
        )
