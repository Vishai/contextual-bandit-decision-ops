from __future__ import annotations

import numpy as np

from .config import SimulationConfig
from .schemas import REGIONS, UserContext


def generate_user_contexts(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> list[UserContext]:
    return [
        UserContext(
            age=float(rng.uniform(18.0, 70.0)),
            engagement=float(rng.uniform(0.0, 1.0)),
            region=REGIONS[int(rng.integers(0, len(REGIONS)))],
        )
        for _ in range(config.n_events)
    ]
