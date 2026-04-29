"""Use case for running genetic optimization from the application layer."""

from __future__ import annotations

from typing import Any

from application.services.genetic_optimization_service import GeneticOptimizationService


class RunGeneticOptimizationUseCase:
    def __init__(self, service: GeneticOptimizationService):
        self.service = service

    def execute(self, **options: Any) -> dict[str, Any]:
        return self.service.run(**options)
