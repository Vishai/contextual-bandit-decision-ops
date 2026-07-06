from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .schemas import EVENT_COLUMNS, BanditEvent


def events_to_frame(events: list[BanditEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        (asdict(event) for event in events),
        columns=EVENT_COLUMNS,
    )


def write_event_log(events: list[BanditEvent], output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = events_to_frame(events)
    frame.to_csv(output_path, index=False)
    return frame
