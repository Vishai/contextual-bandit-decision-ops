from __future__ import annotations

import numpy as np

from .schemas import REGIONS, UserContext


def generate_user_contexts(
    n_events: int,
    rng: np.random.Generator,
) -> list[UserContext]:
    return [
        UserContext(
            age=float(rng.uniform(18.0, 70.0)),
            engagement=float(rng.uniform(0.0, 1.0)),
            region=REGIONS[int(rng.integers(0, len(REGIONS)))],
        )
        for _ in range(n_events)
    ]
