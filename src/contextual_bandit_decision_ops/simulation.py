from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .event_log import write_event_log
from .schemas import REGIONS, BanditEvent, UserContext
from .synthetic import generate_user_contexts
from .validation import validate_event_log


def reward_probability(context: UserContext, action: int) -> float:
    base_probability = 0.08 + (0.25 * context.engagement) + (0.0015 * (context.age - 18.0))
    action_uplift = (
        0.06 * (action == 0 and context.engagement < 0.5)
        + 0.12 * (action == 1 and context.region in {"north", "east"})
        + 0.10 * (action == 2 and context.engagement >= 0.5)
    )
    return float(np.clip(base_probability + action_uplift, 0.01, 0.95))


def simulate_bandit_events(config: SimulationConfig) -> list[BanditEvent]:
    rng = np.random.default_rng(config.seed)
    contexts = generate_user_contexts(config.n_events, rng)
    events: list[BanditEvent] = []
    base_time = config.base_timestamp

    for index, context in enumerate(contexts):
        action = int(rng.integers(0, config.n_actions))
        action_reward_probability = reward_probability(context, action)
        reward = int(rng.random() < action_reward_probability)

        event = BanditEvent(
            event_id=f"event-{config.seed}-{index:06d}",
            user_id=f"user-{index:06d}",
            context_age=context.age,
            context_engagement=context.engagement,
            context_region=context.region,
            action=action,
            reward=reward,
            reward_probability=action_reward_probability,
            propensity=1.0 / config.n_actions,
            timestamp=(base_time + timedelta(minutes=index)).isoformat(),
            seed=config.seed,
        )
        events.append(event)

    validate_event_log(events, config)
    return events


def write_report(config: SimulationConfig, frame: pd.DataFrame) -> Path:
    config.report_md.parent.mkdir(parents=True, exist_ok=True)
    action_counts = frame["action"].value_counts().reindex(range(config.n_actions), fill_value=0)
    reward_rate = float(frame["reward"].mean()) if len(frame) else 0.0

    report_lines = [
        "# Synthetic Bandit Log Summary",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Row count | {len(frame)} |",
        f"| Seed | {config.seed} |",
        f"| Reward rate | {reward_rate:.2%} |",
        "",
        "## Action distribution",
        "",
        "| Action | Count | Share |",
        "| ---: | ---: | ---: |",
    ]
    for action_id in range(config.n_actions):
        action_count = int(action_counts[action_id])
        report_lines.append(f"| {action_id} | {action_count} | {action_count / len(frame):.2%} |")

    report_lines.extend(
        [
            "",
            "## Numeric feature summary",
            "",
            "| Feature | Mean | Minimum | Maximum |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| context_age | {frame['context_age'].mean():.2f} | "
                f"{frame['context_age'].min():.2f} | {frame['context_age'].max():.2f} |"
            ),
            (
                f"| context_engagement | {frame['context_engagement'].mean():.3f} | "
                f"{frame['context_engagement'].min():.3f} | "
                f"{frame['context_engagement'].max():.3f} |"
            ),
            "",
            "## Region distribution",
            "",
            "| Region | Count |",
            "| --- | ---: |",
        ]
    )
    region_counts = frame["context_region"].value_counts()
    for region in REGIONS:
        report_lines.append(f"| {region} | {int(region_counts.get(region, 0))} |")

    config.report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return config.report_md


def generate_synthetic_bandit_log(config: SimulationConfig) -> pd.DataFrame:
    events = simulate_bandit_events(config)
    frame = write_event_log(events, config.output_csv)
    write_report(config, frame)
    return frame
