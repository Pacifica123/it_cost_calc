"""Analytics adapters for SolutionComponent rows.

C6 connects the advanced component editor to AHP, GA and GA+AHP through the
existing ``CandidateConfiguration`` contract.  The service deliberately does not
create a second alternatives format: it normalizes editor rows, filters them by
profile, enriches profile-specific metrics and then exposes the same candidate
configuration payloads and GA item properties that the rest of the application
already understands.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

from application.services.analysis_scope_profile_service import (
    AnalysisScopeProfile,
    AnalysisScopeProfileService,
)
from application.services.candidate_configuration_service import CandidateConfigurationService
from application.services.solution_component_normalization_service import (
    SolutionComponentNormalizationService,
)
from application.services.tco_model_service import TCOModelService
from domain import (
    AnalysisScope,
    CandidateConfiguration,
    CandidateConfigurationSource,
    ComponentType,
    SolutionComponent,
)
from shared.constants import (
    ANALYSIS_SCOPE_SOFTWARE,
    ANALYSIS_SCOPE_TECHNICAL,
    SOLUTION_COMPONENT_ENTITY,
)

_TECHNICAL_COMPONENT_CATEGORY = {
    ComponentType.SERVER: "server",
    ComponentType.WORKSTATION: "client",
    ComponentType.PERIPHERAL: "client",
    ComponentType.NETWORK_DEVICE: "network",
    ComponentType.SUPPORT_SERVICE: "server_administration",
    ComponentType.BACKUP_SERVICE: "backup",
}
_SOFTWARE_COMPONENT_CATEGORY = {
    ComponentType.SOFTWARE_LICENSE: "licenses",
    ComponentType.SOFTWARE_SUBSCRIPTION: "subscription_licenses",
    ComponentType.SOFTWARE_SERVICE: "licenses",
    ComponentType.IMPLEMENTATION_SERVICE: "migration",
    ComponentType.SUPPORT_SERVICE: "labor_costs",
}
_PROFILED_SCOPES = {ANALYSIS_SCOPE_TECHNICAL, ANALYSIS_SCOPE_SOFTWARE}


class SolutionComponentAnalyticsIntegrationService:
    """Prepare SolutionComponent rows for AHP/GA without ad-hoc alternatives.

    Public methods return plain dictionaries where the boundary is a payload
    boundary. Internally every alternative is a ``CandidateConfiguration`` so AHP,
    GA and GA+AHP see the same model as legacy runtime candidates.
    """

    def __init__(
        self,
        *,
        normalization_service: SolutionComponentNormalizationService | None = None,
        profile_service: AnalysisScopeProfileService | None = None,
        candidate_configuration_service: CandidateConfigurationService | None = None,
        tco_model_service: TCOModelService | None = None,
    ):
        self.profile_service = profile_service or AnalysisScopeProfileService()
        self.tco_model_service = tco_model_service or TCOModelService()
        self.normalization_service = normalization_service or SolutionComponentNormalizationService(
            profile_service=self.profile_service,
            tco_model_service=self.tco_model_service,
        )
        self.candidate_configuration_service = (
            candidate_configuration_service
            or CandidateConfigurationService(tco_model_service=self.tco_model_service)
        )

    def build_profile_pool(
        self,
        components: Iterable[SolutionComponent | Mapping[str, Any]],
        *,
        analysis_scope: str | AnalysisScope,
        candidate_prefix: str = "solution_component",
    ) -> dict[str, Any]:
        """Build the C6 profile-specific alternative pool.

        Drafts, blocked rows, mixed bundles and components outside the selected
        profile are returned in ``excluded_components`` with reasons. They never
        silently become zero-valued alternatives.
        """

        profile = self.profile_service.get_profile(self._scope_value(analysis_scope))
        normalized = self.normalization_service.normalize_many(components)
        ready: list[tuple[SolutionComponent, dict[str, Any]]] = []
        excluded: list[dict[str, Any]] = []
        warnings: list[str] = []

        for component in normalized:
            reasons = self._profile_exclusion_reasons(component, profile)
            if reasons:
                excluded.append(self._excluded_row(component, reasons))
                warnings.extend(f"{component.name or component.id}: {reason}" for reason in reasons)
                continue
            payload = self.to_analysis_component_payload(component, profile=profile)
            metric_warnings = payload.get("analysis_warnings", [])
            if metric_warnings:
                warnings.extend(f"{component.name or component.id}: {warning}" for warning in metric_warnings)
            ready.append((component, payload))

        candidate_models = [
            self._candidate_from_component(
                component,
                payload,
                profile=profile,
                candidate_id=f"{candidate_prefix}:{profile.scope}:{component.id}",
            )
            for component, payload in ready
        ]
        candidate_payloads = self.candidate_configuration_service.payload(candidate_models)
        ahp_configurations = self.candidate_configuration_service.to_ahp_configurations(
            candidate_models
        )
        ga_items = [deepcopy(payload) for _, payload in ready]
        return {
            "schema_version": 1,
            "integration_stage": "C6",
            "analysis_scope": profile.scope,
            "analysis_profile": profile.to_dict(),
            "input_component_count": len(normalized),
            "ready_component_count": len(ready),
            "excluded_component_count": len(excluded),
            "candidate_configurations": candidate_payloads,
            "ahp_configurations": ahp_configurations,
            "ga_items": ga_items,
            "excluded_components": excluded,
            "warnings": self._unique(warnings),
        }

    def build_profile_pool_from_entities(
        self,
        entities: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        analysis_scope: str | AnalysisScope,
        candidate_prefix: str = "solution_component",
    ) -> dict[str, Any]:
        """Read ``entities.solution_components`` and build a profile pool."""

        return self.build_profile_pool(
            entities.get(SOLUTION_COMPONENT_ENTITY, []),
            analysis_scope=analysis_scope,
            candidate_prefix=candidate_prefix,
        )

    def build_ga_item_properties(
        self,
        components: Iterable[SolutionComponent | Mapping[str, Any]],
        *,
        analysis_scope: str | AnalysisScope,
    ) -> list[dict[str, Any]]:
        """Return GA item properties for profile-ready editor components."""

        pool = self.build_profile_pool(components, analysis_scope=analysis_scope)
        return [dict(item) for item in pool.get("ga_items", [])]

    def build_candidate_configurations(
        self,
        components: Iterable[SolutionComponent | Mapping[str, Any]],
        *,
        analysis_scope: str | AnalysisScope,
    ) -> list[dict[str, Any]]:
        """Return profile-ready component alternatives as CandidateConfiguration payloads."""

        pool = self.build_profile_pool(components, analysis_scope=analysis_scope)
        return [dict(item) for item in pool.get("candidate_configurations", [])]

    def to_analysis_component_payload(
        self,
        component: SolutionComponent | Mapping[str, Any],
        *,
        profile: AnalysisScopeProfile,
    ) -> dict[str, Any]:
        """Enrich one normalized component with profile metrics used by AHP/GA."""

        normalized = self.normalization_service.normalize(component)
        payload = self.normalization_service.to_component_payload(normalized)
        metrics = deepcopy(normalized.metrics)
        source_category = self._source_category(normalized, profile=profile)
        quantity = self._number(payload.get("quantity"), default=1.0)
        analysis_cost = self._analysis_cost(payload)
        max_power = self._first_number(
            {**metrics, **payload},
            ("max_power_watts", "max_power", "power_watts", "energy"),
            default=0.0,
        )
        ram_gb = self._first_number({**metrics, **payload}, ("ram_gb", "ram_score", "memory_gb"), default=0.0)
        cpu_cores = self._first_number({**metrics, **payload}, ("cpu_cores", "cpu_score", "cores"), default=0.0)
        storage_gb = self._first_number({**metrics, **payload}, ("storage_gb", "disk_gb", "ssd_gb", "hdd_gb"), default=0.0)
        lan_ports = self._first_number({**metrics, **payload}, ("lan_ports", "lan_port_count", "ethernet_ports"), default=0.0)
        lan_speed_mbps = self._first_number({**metrics, **payload}, ("lan_speed_mbps", "ethernet_speed_mbps", "port_speed_mbps"), default=0.0)
        wifi_total_mbps = self._first_number({**metrics, **payload}, ("wifi_total_mbps", "wifi_speed_mbps", "wireless_speed_mbps"), default=0.0)
        ipv6_support = self._bool_or_none(metrics.get("ipv6_support", payload.get("ipv6_support", metrics.get("ipv6"))))
        performance = self._first_number(
            metrics,
            ("perf", "performance_score", "functionality_score", "cpu_score"),
            default=0.0,
        )
        reliability = self._first_number(
            metrics,
            ("reliability", "reliability_score"),
            default=0.0,
        )
        lifespan = self._first_number(metrics, ("lifespan", "lifetime_years"), default=0.0)
        license_units = self._number(metrics.get("license_units"), default=0.0)
        if profile.scope == ANALYSIS_SCOPE_SOFTWARE and license_units <= 0:
            license_units = quantity
        client_seats = self._number(metrics.get("client_seats"), default=0.0)
        if (
            profile.scope == ANALYSIS_SCOPE_TECHNICAL
            and normalized.component_type == ComponentType.WORKSTATION
            and client_seats <= 0
        ):
            client_seats = quantity

        payload.update(
            {
                "id": normalized.id,
                "name": normalized.name,
                "source_format": "solution_component",
                "source_component_id": normalized.id,
                "source_category": source_category,
                "category": source_category,
                "role": "software" if profile.scope == ANALYSIS_SCOPE_SOFTWARE else source_category,
                "profile_scope": profile.scope,
                "profile_label": profile.label,
                "quantity": quantity,
                "price": analysis_cost / quantity if quantity > 0 else analysis_cost,
                "unit_price": analysis_cost / quantity if quantity > 0 else analysis_cost,
                "cost": analysis_cost,
                "total_cost": analysis_cost,
                "capital_cost": analysis_cost,
                "energy": max_power if profile.scope == ANALYSIS_SCOPE_TECHNICAL else 0.0,
                "max_power": max_power,
                "max_power_watts": max_power,
                "ram_gb": ram_gb,
                "cpu_cores": cpu_cores,
                "storage_gb": storage_gb,
                "lan_ports": lan_ports,
                "lan_speed_mbps": lan_speed_mbps,
                "wifi_total_mbps": wifi_total_mbps,
                "ipv6_support": ipv6_support,
                "client_seats": client_seats,
                "license_units": license_units,
                "perf": performance,
                "cpu_score": self._first_number(metrics, ("cpu_score",), default=performance),
                "ram_score": self._first_number(metrics, ("ram_score",), default=0.0),
                "reliability": reliability,
                "lifespan": lifespan,
                "functionality_score": self._first_number(metrics, ("functionality_score",), default=performance),
                "support_score": self._first_number(metrics, ("support_score",), default=0.0),
                "selected_count": 1.0,
                "analysis_warnings": self._metric_warnings(
                    normalized,
                    profile=profile,
                    source_category=source_category,
                    analysis_cost=analysis_cost,
                    max_power=max_power,
                    ram_gb=ram_gb,
                    cpu_cores=cpu_cores,
                    storage_gb=storage_gb,
                    lan_ports=lan_ports,
                    lan_speed_mbps=lan_speed_mbps,
                    wifi_total_mbps=wifi_total_mbps,
                    ipv6_support=ipv6_support,
                    client_seats=client_seats,
                    license_units=license_units,
                ),
            }
        )
        analysis_metadata = dict(payload.get("metadata") or {})
        analysis_metadata.setdefault("analytics_integration", {})
        analysis_metadata["analytics_integration"].update(
            {
                "stage": "C6",
                "through": "CandidateConfiguration",
                "profile_scope": profile.scope,
                "source_component_id": normalized.id,
            }
        )
        payload["metadata"] = analysis_metadata
        return payload

    def _candidate_from_component(
        self,
        component: SolutionComponent,
        payload: Mapping[str, Any],
        *,
        profile: AnalysisScopeProfile,
        candidate_id: str,
    ) -> CandidateConfiguration:
        totals = self._candidate_totals(payload)
        metrics = self._candidate_metrics(payload, profile=profile)
        candidate = CandidateConfiguration(
            id=candidate_id,
            name=component.name,
            scope=AnalysisScope(profile.scope),
            components=[deepcopy(dict(payload))],
            totals=totals,
            metrics=metrics,
            source=CandidateConfigurationSource.MANUAL,
            metadata={
                "source_format": "solution_component_analytics_alternative",
                "integration_stage": "C6",
                "through": "CandidateConfiguration",
                "analysis_profile": profile.to_dict(),
                "source_component_ids": [component.id],
                "metric_warnings": list(payload.get("analysis_warnings", [])),
            },
        )
        return self.tco_model_service.attach_to_candidate(candidate)

    def _profile_exclusion_reasons(
        self,
        component: SolutionComponent,
        profile: AnalysisScopeProfile,
    ) -> list[str]:
        reasons: list[str] = []
        if not component.candidate_eligible:
            reasons.extend(component.blocking_errors)
            reasons.extend(component.validation_warnings)
            if not reasons:
                reasons.append("component is not candidate_eligible")
            return self._unique(reasons)
        if component.scope is None:
            return ["component.scope is required for profile analytics"]
        if component.scope.value not in _PROFILED_SCOPES:
            reasons.append(
                "mixed/common component requires an explicit split or summary profile before AHP/GA."
            )
        elif component.scope.value != profile.scope:
            reasons.append(
                f"component scope {component.scope.value!r} does not match analysis profile {profile.scope!r}."
            )
        if component.component_type is None:
            reasons.append("component.component_type is required for profile analytics")
        elif component.component_type.value not in profile.component_types:
            reasons.append(
                f"component_type {component.component_type.value!r} is not allowed for profile {profile.scope!r}."
            )
        return self._unique(reasons)

    def _excluded_row(self, component: SolutionComponent, reasons: Sequence[str]) -> dict[str, Any]:
        return {
            "id": component.id,
            "name": component.name,
            "scope": component.scope.value if component.scope else None,
            "component_type": component.component_type.value if component.component_type else None,
            "normalization_state": component.normalization_state.value,
            "candidate_eligible": component.candidate_eligible,
            "reasons": list(reasons),
        }

    def _source_category(self, component: SolutionComponent, *, profile: AnalysisScopeProfile) -> str:
        metrics = component.metrics if isinstance(component.metrics, Mapping) else {}
        explicit = metrics.get("source_category") or metrics.get("category")
        if explicit:
            return str(explicit)
        if component.component_type is None:
            return "unknown"
        if profile.scope == ANALYSIS_SCOPE_TECHNICAL:
            return _TECHNICAL_COMPONENT_CATEGORY.get(component.component_type, "unknown")
        return _SOFTWARE_COMPONENT_CATEGORY.get(component.component_type, "licenses")

    def _analysis_cost(self, payload: Mapping[str, Any]) -> float:
        purchase_cost = self._number(payload.get("purchase_cost"), default=0.0)
        one_time = self._number(payload.get("one_time_cost"), default=0.0)
        monthly = self._number(payload.get("recurring_monthly_cost"), default=0.0)
        energy = self._number(payload.get("energy_cost"), default=0.0)
        # One year is enough for analytics ranking: full TCO is still attached in totals.tco.
        return purchase_cost + one_time + monthly * 12.0 + energy * 12.0

    def _candidate_totals(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        total_cost = self._number(payload.get("total_cost"), default=0.0)
        energy = self._number(payload.get("energy"), default=0.0)
        return {
            "capital_cost": total_cost,
            "total_cost": total_cost,
            "total_energy": energy,
            "energy": energy,
            "total_power_watts": energy * self._number(payload.get("quantity"), default=1.0),
            "total_ram_gb": self._number(payload.get("ram_gb"), default=0.0) * self._number(payload.get("quantity"), default=1.0),
            "total_cpu_cores": self._number(payload.get("cpu_cores"), default=0.0) * self._number(payload.get("quantity"), default=1.0),
            "total_storage_gb": self._number(payload.get("storage_gb"), default=0.0) * self._number(payload.get("quantity"), default=1.0),
            "lan_ports": self._number(payload.get("lan_ports"), default=0.0) * self._number(payload.get("quantity"), default=1.0),
            "lan_speed_mbps": self._number(payload.get("lan_speed_mbps"), default=0.0),
            "wifi_total_mbps": self._number(payload.get("wifi_total_mbps"), default=0.0) * self._number(payload.get("quantity"), default=1.0),
            "ipv6_support_count": self._bool_as_number(payload.get("ipv6_support")) * self._number(payload.get("quantity"), default=1.0),
            "metric_warnings": list(payload.get("analysis_warnings", [])),
            "component_count": 1,
            "solution_component_count": 1,
        }

    def _candidate_metrics(self, payload: Mapping[str, Any], *, profile: AnalysisScopeProfile) -> dict[str, Any]:
        return {
            "profile_scope": profile.scope,
            "component_count": 1,
            "solution_component_count": 1,
            "total_performance": self._number(payload.get("perf"), default=0.0)
            + self._number(payload.get("cpu_score"), default=0.0)
            + self._number(payload.get("ram_score"), default=0.0),
            "avg_reliability": self._number(payload.get("reliability"), default=0.0),
            "lifespan": self._number(payload.get("lifespan"), default=0.0),
            "client_capacity": self._number(payload.get("client_seats"), default=0.0),
            "software_license_quantity": self._number(payload.get("license_units"), default=0.0),
            "total_power_watts": self._number(payload.get("max_power_watts", payload.get("max_power")), default=0.0)
            * self._number(payload.get("quantity"), default=1.0),
            "total_ram_gb": self._number(payload.get("ram_gb"), default=0.0)
            * self._number(payload.get("quantity"), default=1.0),
            "total_cpu_cores": self._number(payload.get("cpu_cores"), default=0.0)
            * self._number(payload.get("quantity"), default=1.0),
            "total_storage_gb": self._number(payload.get("storage_gb"), default=0.0)
            * self._number(payload.get("quantity"), default=1.0),
            "lan_ports": self._number(payload.get("lan_ports"), default=0.0)
            * self._number(payload.get("quantity"), default=1.0),
            "lan_speed_mbps": self._number(payload.get("lan_speed_mbps"), default=0.0),
            "wifi_total_mbps": self._number(payload.get("wifi_total_mbps"), default=0.0)
            * self._number(payload.get("quantity"), default=1.0),
            "ipv6_support_count": self._bool_as_number(payload.get("ipv6_support"))
            * self._number(payload.get("quantity"), default=1.0),
            "metric_warnings": list(payload.get("analysis_warnings", [])),
            "people": int(self._number(payload.get("client_seats"), default=0.0)),
        }

    def _metric_warnings(
        self,
        component: SolutionComponent,
        *,
        profile: AnalysisScopeProfile,
        source_category: str,
        analysis_cost: float,
        max_power: float,
        ram_gb: float,
        cpu_cores: float,
        storage_gb: float,
        lan_ports: float,
        lan_speed_mbps: float,
        wifi_total_mbps: float,
        ipv6_support: bool | None,
        client_seats: float,
        license_units: float,
    ) -> list[str]:
        warnings: list[str] = []
        if not source_category or source_category == "unknown":
            warnings.append("source_category is missing; profile coverage may be ambiguous")
        if analysis_cost <= 0:
            warnings.append("capital_cost metric is missing; component will not be cost-comparable")
        if profile.scope == ANALYSIS_SCOPE_TECHNICAL:
            if component.component_type == ComponentType.WORKSTATION and client_seats <= 0:
                warnings.append("workstation has no client_seats; it cannot cover user workplaces")
            if component.component_type in {ComponentType.SERVER, ComponentType.WORKSTATION, ComponentType.NETWORK_DEVICE} and max_power <= 0:
                warnings.append("technical component has no max_power_watts/max_power; power criterion will be incomplete")
            if component.component_type in {ComponentType.SERVER, ComponentType.WORKSTATION}:
                if ram_gb <= 0:
                    warnings.append("compute metric ram_gb is missing")
                if cpu_cores <= 0:
                    warnings.append("compute metric cpu_cores is missing")
                if storage_gb <= 0:
                    warnings.append("compute metric storage_gb is missing")
            if component.component_type == ComponentType.NETWORK_DEVICE:
                if lan_ports <= 0:
                    warnings.append("network metric lan_ports is missing")
                if lan_speed_mbps <= 0:
                    warnings.append("network metric lan_speed_mbps is missing")
                if wifi_total_mbps <= 0:
                    warnings.append("network metric wifi_total_mbps is missing")
                if ipv6_support is None:
                    warnings.append("network metric ipv6_support is missing")
        if profile.scope == ANALYSIS_SCOPE_SOFTWARE and license_units <= 0:
            warnings.append("software component has no license_units/quantity metric")
        if not any(
            self._number(component.metrics.get(key), default=0.0) > 0
            for key in ("perf", "performance_score", "functionality_score", "cpu_score")
        ):
            warnings.append("quality/performance metric is missing; AHP/GA will not infer a hidden score")
        return self._unique(warnings)

    def _bool_or_none(self, value: Any) -> bool | None:
        if self._is_blank(value):
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "да", "+", "есть", "поддерживается"}:
            return True
        if text in {"0", "false", "no", "n", "нет", "-", "не поддерживается"}:
            return False
        return None

    def _bool_as_number(self, value: Any) -> float:
        return 1.0 if self._bool_or_none(value) is True else 0.0

    def _scope_value(self, value: str | AnalysisScope) -> str:
        return value.value if isinstance(value, AnalysisScope) else str(value)

    def _first_number(
        self,
        mapping: Mapping[str, Any],
        keys: Sequence[str],
        *,
        default: float,
    ) -> float:
        for key in keys:
            if key not in mapping:
                continue
            raw_value = mapping.get(key)
            if self._is_blank(raw_value) or self._is_structured_value(raw_value):
                continue
            value = self._number(raw_value, default=default)
            return value
        return float(default)

    def _number(self, value: Any, *, default: float) -> float:
        try:
            if self._is_blank(value):
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _is_blank(self, value: Any) -> bool:
        return value is None or value == ""

    def _is_structured_value(self, value: Any) -> bool:
        return isinstance(value, (Mapping, list, tuple, set))

    def _unique(self, values: Sequence[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value)
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result


__all__ = ["SolutionComponentAnalyticsIntegrationService"]
