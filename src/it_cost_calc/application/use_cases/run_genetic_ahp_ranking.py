"""Use case for the combined GA → AHP decision scenario."""

from __future__ import annotations

from typing import Any

from application.services.genetic_ahp_ranking_service import GeneticAhpRankingService


class RunGeneticAhpRankingUseCase:
    def __init__(self, service: GeneticAhpRankingService):
        self.service = service

    def execute(self, **options: Any) -> dict[str, Any]:
        return self.service.run(**options)


__all__ = ["RunGeneticAhpRankingUseCase"]
