"""Test configuration and helpers for the evaluation sandbox."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

# Ensure the repository root and ``src`` directory are available on the Python path
# even when pytest is invoked from outside the project or without installing the
# package in editable mode.
for candidate in (PROJECT_ROOT, SRC_PATH):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
