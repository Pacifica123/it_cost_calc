from __future__ import annotations

import math
from typing import Dict


class Item:
    """
    Представление предмета как контейнера произвольных числовых свойств.
    Пример: Item(weight=3, CPU=4, RAM=2, energy=5)
    """

    properties: Dict[str, float]

    def __init__(self, **properties):
        # В dataclass frozen - делаем через object.__setattr__
        object.__setattr__(self, "properties", dict(properties))


class FuzzyValue:
    low: float
    high: float


class FuzzyTerm:
    low: float
    high: float
    a1: float = 0.2  # параметр эксперта

    def mu_true(self, x: float) -> float:
        """Функция истинности."""
        if x < self.a1 or x > 1.0:
            return max(
                0.0,
                min(
                    1.0,
                    0.5 * (1.0 + (math.pi / 2.0) * ((2.0 * x - 1.0 - self.a1) / (1.0 - self.a1))),
                ),
            )
        v = 0.5 * (1.0 + (math.pi / 2.0) * ((2.0 * x - 1.0 - self.a1) / (1.0 - self.a1)))
        return max(0.0, min(1.0, v))

    def mu_false(self, x: float) -> float:
        if x < 0.0 or x > 1.0 - self.a1:
            return max(
                0.0,
                min(
                    1.0,
                    0.5 * (1.0 + (math.pi / 2.0) * ((1.0 - self.a1 - 2.0 * x) / (1.0 - self.a1))),
                ),
            )
        v = 0.5 * (1.0 + (math.pi / 2.0) * ((1.0 - self.a1 - 2.0 * x) / (1.0 - self.a1)))
        return max(0.0, min(1.0, v))

    def centroid(self) -> float:
        """Простая дефаззификация по центру интервала."""
        return (self.low + self.high) / 2.0
