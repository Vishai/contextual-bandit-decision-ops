from pathlib import Path

import pandas as pd
import pytest

from contextual_bandit_decision_ops.cli import main
from contextual_bandit_decision_ops.config import SimulationConfig
from contextual_bandit_decision_ops.schemas import EVENT_COLUMNS
from contextual_bandit_decision_ops.simulation import generate_synthetic_bandit_log, simulate_bandit_events
from contextual_bandit_decision_ops.validation import validate_bandit_event


def test_generation_is_byte_for_byte_deterministic(tmp_path: Path) -> None:
    config_a = SimulationConfig(
        n_events=12,
        seed=7,
        output_csv=tmp_path / "a.csv",
        report_md=tmp_path / "a.md",
    )
    config_b = SimulationConfig(
        n_events=12,
        seed=7,
        output_csv=tmp_path / "b.csv",
        report_md=tmp_path / "b.md",
    )

    df_a = generate_synthetic_bandit_log(config_a)
    df_b = generate_synthetic_bandit_log(config_b)

    assert df_a.equals(df_b)
    assert config_a.output_csv.read_bytes() == config_b.output_csv.read_bytes()
    assert config_a.report_md.read_bytes() == config_b.report_md.read_bytes()


def test_csv_schema_is_complete_and_valid(tmp_path: Path) -> None:
    config = SimulationConfig(
        n_events=20,
        seed=11,
        output_csv=tmp_path / "events.csv",
        report_md=tmp_path / "summary.md",
    )
    generate_synthetic_bandit_log(config)
    frame = pd.read_csv(config.output_csv)

    assert tuple(frame.columns) == EVENT_COLUMNS
    assert len(frame) == config.n_events
    assert not frame.isna().any().any()
    assert frame["event_id"].is_unique
    assert pd.to_datetime(frame["timestamp"], utc=True, errors="raise").notna().all()


def test_event_values_and_logging_propensity_are_valid() -> None:
    config = SimulationConfig(n_events=30, seed=11, n_actions=3)
    events = simulate_bandit_events(config)

    assert len(events) == config.n_events
    for event in events:
        validate_bandit_event(event, config)
        assert event.action in range(config.n_actions)
        assert event.reward in {0, 1}
        assert 0.0 <= event.reward_probability <= 1.0
        assert event.propensity == pytest.approx(1 / config.n_actions)
        assert 18.0 <= event.context_age <= 70.0
        assert 0.0 <= event.context_engagement <= 1.0


def test_report_creation(tmp_path: Path) -> None:
    config = SimulationConfig(
        n_events=10,
        seed=3,
        output_csv=tmp_path / "synthetic.csv",
        report_md=tmp_path / "summary.md",
    )

    generate_synthetic_bandit_log(config)

    report_text = config.report_md.read_text(encoding="utf-8")
    assert "# Synthetic Bandit Log Summary" in report_text
    assert "| Row count | 10 |" in report_text
    assert "## Action distribution" in report_text
    assert "Reward rate" in report_text
    assert "## Numeric feature summary" in report_text
    assert "## Region distribution" in report_text


def test_cli_writes_default_output_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    output_csv = Path("data/processed/synthetic_bandit_log.csv")
    report_md = Path("reports/synthetic_bandit_log_summary.md")

    assert main([]) == 0
    assert output_csv.exists()
    assert report_md.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_events", 0),
        ("n_actions", 0),
        ("seed", -1),
    ],
)
def test_invalid_config_is_rejected(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        SimulationConfig(**{field: value})
