"""Lightweight per-scope candidate pool for decision-method workspaces.

The service is deliberately small: it does not persist history, does not start
background recalculations and does not replace GA/AHP/Pareto services.  It only
keeps the last explicit pool of alternatives for one analysis scope and adapts
that pool to the legacy AHP and criteria-importance UI formats.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from application.services.analysis_scope_profile_service import AnalysisScopeProfileService
from application.services.candidate_configuration_service import CandidateConfigurationService
from shared.constants import ANALYSIS_SCOPE_TECHNICAL


@dataclass(slots=True)
class CandidatePoolSnapshot:
    """Serializable snapshot of alternatives for one ПО/ТО analysis scope."""

    scope: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    source_label: str = "ручной пул альтернатив"
    source_method: str = "manual"

    def summary(self) -> str:
        count = len(self.candidates)
        if count == 0:
            return "Общий пул альтернатив пуст."
        return f"Общий пул: {count} альтернатив; источник: {self.source_label}."


class ScopedCandidatePoolService:
    """Stores the current candidate alternatives separately for each analysis scope."""

    def __init__(
        self,
        *,
        candidate_configuration_service: CandidateConfigurationService | None = None,
        profile_service: AnalysisScopeProfileService | None = None,
    ) -> None:
        self.candidate_configuration_service = (
            candidate_configuration_service or CandidateConfigurationService()
        )
        self.profile_service = profile_service or AnalysisScopeProfileService()
        self._snapshots: dict[str, CandidatePoolSnapshot] = {}

    def replace(
        self,
        scope: str | None,
        candidates: Sequence[Mapping[str, Any]],
        *,
        source_label: str = "ручной пул альтернатив",
        source_method: str = "manual",
    ) -> CandidatePoolSnapshot:
        scope_key = self._scope(scope)
        payload = self._deduplicated_payload(candidates)
        snapshot = CandidatePoolSnapshot(
            scope=scope_key,
            candidates=payload,
            source_label=source_label,
            source_method=source_method,
        )
        self._snapshots[scope_key] = snapshot
        return snapshot

    def replace_from_ga_result(
        self,
        scope: str | None,
        result: Mapping[str, Any],
        *,
        source_label: str = "top-N GA-кандидатов",
    ) -> CandidatePoolSnapshot:
        candidates = self._candidate_configurations_from_result(result)
        return self.replace(
            scope,
            candidates,
            source_label=source_label,
            source_method="ga",
        )

    def get(self, scope: str | None) -> CandidatePoolSnapshot:
        scope_key = self._scope(scope)
        return self._snapshots.get(scope_key) or CandidatePoolSnapshot(scope=scope_key)

    def clear(self, scope: str | None) -> CandidatePoolSnapshot:
        scope_key = self._scope(scope)
        snapshot = CandidatePoolSnapshot(scope=scope_key)
        self._snapshots[scope_key] = snapshot
        return snapshot

    def to_ahp_configurations(self, scope: str | None) -> list[dict[str, Any]]:
        snapshot = self.get(scope)
        return self.candidate_configuration_service.to_ahp_configurations(snapshot.candidates)

    def to_criteria_case(self, scope: str | None) -> dict[str, Any]:
        """Build a simple Pareto/criteria-importance case from the shared pool.

        The legacy criteria-importance module expects all criteria to be
        maximized.  Profile criteria with ``direction='min'`` are therefore
        converted to a 1..5 utility scale where lower raw values receive better
        scores.  This keeps Pareto direction explicit without adding a complex
        rule engine to the MVP.
        """

        scope_key = self._scope(scope)
        snapshot = self.get(scope_key)
        profile = self.profile_service.get_profile(scope_key)
        criteria = [
            {
                "id": criterion.id,
                "name": criterion.label,
                "direction": criterion.direction,
                "metric": criterion.metric or criterion.id,
            }
            for criterion in profile.criteria
        ]
        alternatives = [
            {"id": str(candidate.get("id") or f"candidate_{index}"), "name": str(candidate.get("name") or f"Альтернатива {index}")}
            for index, candidate in enumerate(snapshot.candidates, start=1)
        ]
        raw_by_criterion = {
            criterion["id"]: [
                self._criterion_raw_value(candidate, criterion)
                for candidate in snapshot.candidates
            ]
            for criterion in criteria
        }
        scores: dict[str, dict[str, float]] = {}
        for alt_index, alternative in enumerate(alternatives):
            alt_scores: dict[str, float] = {}
            for criterion in criteria:
                values = raw_by_criterion[criterion["id"]]
                raw_value = values[alt_index]
                alt_scores[criterion["id"]] = self._utility_score(
                    raw_value,
                    values,
                    direction=str(criterion.get("direction") or "max"),
                )
            scores[alternative["id"]] = alt_scores

        return {
            "name": f"Pareto-кейс из общего пула {profile.label}",
            "criteria": criteria,
            "alternatives": alternatives,
            "scores": scores,
            "relations": [],
            "candidate_configurations": deepcopy(snapshot.candidates),
            "metadata": {
                "analysis_scope": scope_key,
                "candidate_pool_source": snapshot.source_label,
                "source_method": snapshot.source_method,
                "score_scale": "1..5 utility, min criteria inverted",
            },
        }

    def _deduplicated_payload(self, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        payload = self.candidate_configuration_service.payload(candidates)
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, candidate in enumerate(payload, start=1):
            candidate_id = str(candidate.get("id") or f"candidate_{index}")
            if candidate_id in seen:
                candidate_id = f"{candidate_id}_{index}"
                candidate["id"] = candidate_id
            seen.add(candidate_id)
            result.append(candidate)
        return result

    def _candidate_configurations_from_result(self, result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        direct = result.get("candidate_configurations")
        if isinstance(direct, list):
            return [item for item in direct if isinstance(item, Mapping)]
        genetic = result.get("genetic_optimization")
        if isinstance(genetic, Mapping):
            nested = genetic.get("candidate_configurations")
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, Mapping)]
        export_payload = result.get("export_payload")
        if isinstance(export_payload, Mapping):
            nested = export_payload.get("candidate_configurations")
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, Mapping)]
        return []

    def _criterion_raw_value(self, candidate: Mapping[str, Any], criterion: Mapping[str, Any]) -> float:
        criterion_id = str(criterion.get("id") or "")
        metric = str(criterion.get("metric") or criterion_id)
        metrics = candidate.get("metrics", {}) if isinstance(candidate.get("metrics"), Mapping) else {}
        totals = candidate.get("totals", {}) if isinstance(candidate.get("totals"), Mapping) else {}

        for container in (metrics, totals):
            if criterion_id in container:
                return self._number(container.get(criterion_id))
            if metric in container:
                return self._number(container.get(metric))

        raw_scores = metrics.get("raw_scores_by_criterion")
        if isinstance(raw_scores, Mapping):
            if criterion_id in raw_scores:
                return self._number(raw_scores.get(criterion_id))
            if metric in raw_scores:
                return self._number(raw_scores.get(metric))

        aliases = {
            "total_ram_gb": ("ram_gb",),
            "total_cpu_cores": ("cpu_cores",),
            "total_storage_gb": ("storage_gb",),
            "total_power_watts": ("total_power_watts", "power_watts", "max_power", "energy"),
            "capital_cost": ("capital_cost", "total_cost", "tco"),
            "software_license_quantity": ("software_license_quantity", "license_units"),
        }
        for alias in aliases.get(metric, ()):  # noqa: SIM118 - explicit readable fallback chain
            if alias in totals:
                return self._number(totals.get(alias))
            if alias in metrics:
                return self._number(metrics.get(alias))
            if isinstance(raw_scores, Mapping) and alias in raw_scores:
                return self._number(raw_scores.get(alias))
        return 0.0

    def _utility_score(self, value: float, values: Sequence[float], *, direction: str) -> float:
        if not values:
            return 3.0
        minimum = min(values)
        maximum = max(values)
        if abs(maximum - minimum) < 1e-12:
            return 3.0
        ratio = (float(value) - minimum) / (maximum - minimum)
        if direction == "min":
            ratio = 1.0 - ratio
        return 1.0 + 4.0 * max(0.0, min(1.0, ratio))

    def _scope(self, scope: str | None) -> str:
        profile = self.profile_service.get_profile(scope or ANALYSIS_SCOPE_TECHNICAL)
        return profile.scope

    def _number(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


__all__ = ["CandidatePoolSnapshot", "ScopedCandidatePoolService"]
