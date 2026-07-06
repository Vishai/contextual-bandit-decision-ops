from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalEventLogger:
    """Append decision and feedback records to local JSONL files."""

    def __init__(self, decision_path: Path, feedback_path: Path) -> None:
        self.decision_path = decision_path
        self.feedback_path = feedback_path

    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def log_decision(self, record: dict[str, Any]) -> None:
        self._append(self.decision_path, record)

    def log_feedback(self, record: dict[str, Any]) -> None:
        self._append(self.feedback_path, record)
