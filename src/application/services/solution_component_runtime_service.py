"""Runtime storage helpers for SolutionComponent rows.

The service keeps the advanced component editor data in a dedicated runtime
section.  It does not require a migration of legacy CAPEX/OPEX rows and it never
lets draft or blocked components silently become analytical input.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from application.ports import EntityRepository
from application.services.solution_component_normalization_service import (
    SolutionComponentNormalizationService,
)
from domain import SolutionComponent, ensure_solution_component, to_plain_data
from shared.constants import SOLUTION_COMPONENT_ENTITY, SOLUTION_COMPONENT_SCHEMA_VERSION


class SolutionComponentRuntimeService:
    """Persist and read SolutionComponent rows from the runtime repository."""

    ENTITY_NAME = SOLUTION_COMPONENT_ENTITY
    SCHEMA_VERSION = SOLUTION_COMPONENT_SCHEMA_VERSION

    def __init__(
        self,
        repository: EntityRepository,
        *,
        normalization_service: SolutionComponentNormalizationService | None = None,
    ):
        self.repository = repository
        self.normalization_service = normalization_service or SolutionComponentNormalizationService()

    def list_rows(self) -> list[dict[str, Any]]:
        """Return raw runtime rows from the dedicated section."""

        return [deepcopy(dict(row)) for row in self.repository.list(self.ENTITY_NAME)]

    def list_components(self) -> list[SolutionComponent]:
        """Return all stored components after current normalization rules."""

        return self.normalization_service.normalize_many(self.list_rows())

    def list_strict_components(self) -> list[SolutionComponent]:
        """Return components that can safely participate in strict analytics."""

        return [component for component in self.list_components() if component.candidate_eligible]

    def list_draft_components(self) -> list[SolutionComponent]:
        """Return components saved as draft, blocked or otherwise excluded."""

        return [component for component in self.list_components() if not component.candidate_eligible]

    def add_component(
        self,
        component: SolutionComponent | Mapping[str, Any],
    ) -> SolutionComponent:
        """Normalize, stamp and persist one component."""

        normalized = self._normalized(component)
        self.repository.add(self.ENTITY_NAME, self._storage_row(normalized))
        return normalized

    def update_component(
        self,
        index: int,
        component: SolutionComponent | Mapping[str, Any],
    ) -> SolutionComponent:
        """Replace a stored component while preserving the runtime section contract."""

        normalized = self._normalized(component)
        self.repository.update(self.ENTITY_NAME, index, self._storage_row(normalized))
        return normalized

    def delete_component(self, index: int) -> SolutionComponent:
        """Delete and return one stored component after normalization."""

        row = self.repository.delete(self.ENTITY_NAME, index)
        return self._normalized(row)

    def replace_components(
        self,
        components: Sequence[SolutionComponent | Mapping[str, Any]],
    ) -> list[SolutionComponent]:
        """Replace only SolutionComponent rows, leaving legacy entities intact."""

        while self.repository.list(self.ENTITY_NAME):
            self.repository.delete(self.ENTITY_NAME, 0)
        return [self.add_component(component) for component in components]

    def build_candidate_configuration(self, *, candidate_id: str = "solution_components") -> dict[str, Any]:
        """Build a candidate from strict components and exclude drafts explicitly."""

        candidate = self.normalization_service.to_candidate_configuration(
            self.list_rows(),
            candidate_id=candidate_id,
            name="Компоненты редактора решения",
        )
        return candidate.to_dict()

    def export_snapshot(self) -> dict[str, Any]:
        """Return a user-facing snapshot for export/report checks.

        The payload intentionally avoids debug-only internals.  It keeps the
        schema version, readiness flags, warnings and exclusion reasons because
        these are meaningful to a user and to a diploma report.
        """

        components = self.list_components()
        rows = [self._export_row(component) for component in components]
        return {
            "schema_version": self.SCHEMA_VERSION,
            "entity_name": self.ENTITY_NAME,
            "component_count": len(rows),
            "strict_component_count": sum(1 for row in rows if row["candidate_eligible"]),
            "draft_component_count": sum(1 for row in rows if not row["candidate_eligible"]),
            "components": rows,
        }

    def _normalized(self, component: SolutionComponent | Mapping[str, Any]) -> SolutionComponent:
        return self.normalization_service.normalize(ensure_solution_component(deepcopy(component)))

    def _storage_row(self, component: SolutionComponent) -> dict[str, Any]:
        row = to_plain_data(component)
        row["schema_version"] = self.SCHEMA_VERSION
        row["storage_entity"] = self.ENTITY_NAME
        row["editor_status"] = "strict" if component.candidate_eligible else "draft"
        row["excluded_from_analysis_reason"] = self._exclusion_reason(component)
        metadata = dict(row.get("metadata") or {})
        metadata.setdefault("runtime_storage", {})
        metadata["runtime_storage"].update(
            {
                "schema_version": self.SCHEMA_VERSION,
                "entity_name": self.ENTITY_NAME,
                "format": "solution_component",
            }
        )
        row["metadata"] = metadata
        return row

    def _export_row(self, component: SolutionComponent) -> dict[str, Any]:
        return {
            "id": component.id,
            "name": component.name,
            "scope": component.scope.value if component.scope else None,
            "component_type": component.component_type.value if component.component_type else None,
            "origin": component.origin.value,
            "normalization_state": component.normalization_state.value,
            "editor_status": "strict" if component.candidate_eligible else "draft",
            "candidate_eligible": component.candidate_eligible,
            "tco_eligible": component.tco_eligible,
            "energy_eligible": component.energy_eligible,
            "npv_eligible": component.npv_eligible,
            "analysis_ready": component.analysis_ready,
            "validation_warnings": list(component.validation_warnings),
            "blocking_errors": list(component.blocking_errors),
            "excluded_from_analysis_reason": self._exclusion_reason(component),
        }

    def _exclusion_reason(self, component: SolutionComponent) -> str:
        if component.candidate_eligible:
            return ""
        reasons = [*component.blocking_errors, *component.validation_warnings]
        return "; ".join(dict.fromkeys(str(reason) for reason in reasons if reason))


__all__ = ["SolutionComponentRuntimeService"]
