from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .bayesian_policies import BayesianLogisticThompsonSampling
from .offers import OfferCatalog

ARTIFACT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PolicyArtifact:
    policy: BayesianLogisticThompsonSampling
    model_version: str
    created_at: str
    metadata: dict[str, Any]


def save_policy_artifact(
    policy: BayesianLogisticThompsonSampling,
    path: Path | str,
    model_version: str,
    metadata: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> Path:
    """Write the posterior to JSON. The temp-file-then-rename keeps a reader from
    ever seeing a half-written model."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model_version": model_version,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "metadata": metadata or {},
        "policy": policy.to_dict(),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, target)
    return target


def policy_artifact_from_payload(
    payload: dict[str, Any],
    catalog: OfferCatalog,
) -> PolicyArtifact:
    if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError("unsupported model artifact format")
    return PolicyArtifact(
        policy=BayesianLogisticThompsonSampling.from_dict(payload["policy"], catalog),
        model_version=str(payload["model_version"]),
        created_at=str(payload["created_at"]),
        metadata=dict(payload.get("metadata", {})),
    )


def load_policy_artifact(path: Path | str, catalog: OfferCatalog) -> PolicyArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return policy_artifact_from_payload(payload, catalog)
