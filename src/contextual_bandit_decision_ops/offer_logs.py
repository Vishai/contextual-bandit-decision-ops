from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DECISIONS_FILE = "decisions.jsonl"
FEEDBACK_FILE = "feedback.jsonl"


def log_files(path: Path | None, file_name: str) -> list[Path]:
    """A single log file, or every matching file under a folder (one per service replica)."""
    if path is None:
        return []
    if path.is_dir():
        return sorted(path.rglob(file_name))
    return [path] if path.exists() else []


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_service_logs(
    decision_log: Path | None,
    feedback_log: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions = [
        record for path in log_files(decision_log, DECISIONS_FILE) for record in read_jsonl(path)
    ]
    feedback = [
        record for path in log_files(feedback_log, FEEDBACK_FILE) for record in read_jsonl(path)
    ]
    return decisions, feedback
