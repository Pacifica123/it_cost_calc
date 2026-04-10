from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FuzzyInterval:
    low: float
    high: float

    def centroid(self) -> float:
        return 0.5 * (self.low + self.high)


RI_TABLE = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
    11: 1.51,
    12: 1.48,
}
