from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .bayesian_policies import BayesianLogisticThompsonSampling
from .model_store import PolicyArtifact, load_policy_artifact, save_policy_artifact
from .offers import OfferCatalog


class ModelRegistry:
    """Versioned model artifacts on disk.

    Every promoted model is kept under ``versions/``, ``latest.json`` always holds
    the one the service should serve, and ``registry.jsonl`` records every
    retraining decision, promoted or not.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.latest_path = self.root / "latest.json"
        self.log_path = self.root / "registry.jsonl"

    def current(self, catalog: OfferCatalog) -> PolicyArtifact | None:
        if not self.latest_path.exists():
            return None
        return load_policy_artifact(self.latest_path, catalog)

    def promote(
        self,
        policy: BayesianLogisticThompsonSampling,
        model_version: str,
        metadata: dict[str, Any],
        created_at: datetime | None = None,
    ) -> Path:
        version_path = self.versions_dir / f"{model_version}.json"
        save_policy_artifact(policy, version_path, model_version, metadata, created_at)
        temporary = self.latest_path.with_suffix(".json.tmp")
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(version_path, temporary)
        temporary.replace(self.latest_path)
        self._append({"status": "promoted", "model_version": model_version, **metadata})
        return version_path

    def record(self, status: str, details: dict[str, Any]) -> None:
        self._append({"status": status, **details})

    def history(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _append(self, entry: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        stamped = {"recorded_at": datetime.now(UTC).isoformat(), **entry}
        with self.log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(stamped, sort_keys=True, default=str) + "\n")
