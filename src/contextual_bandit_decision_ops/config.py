from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class SimulationConfig:
    n_events: int = 100
    seed: int = 42
    n_actions: int = 3
    output_csv: Path | str = Path("data/processed/synthetic_bandit_log.csv")
    report_md: Path | str = Path("reports/synthetic_bandit_log_summary.md")
    base_timestamp: datetime = datetime(2024, 1, 1, tzinfo=UTC)

    def __post_init__(self) -> None:
        if self.n_events <= 0:
            raise ValueError("n_events must be positive")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.base_timestamp.tzinfo is None:
            raise ValueError("base_timestamp must include a timezone")
        object.__setattr__(self, "output_csv", Path(self.output_csv))
        object.__setattr__(self, "report_md", Path(self.report_md))
