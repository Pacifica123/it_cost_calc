"""Normalization of runtime CAPEX/OPEX entities during the scope transition.

The project still stores user data grouped by legacy category names.  This
service is the single application-level adapter that derives missing
``scope`` and ``component_type`` values from those categories until the runtime
format becomes explicit everywhere.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from domain import AnalysisScope, ComponentType
from shared.constants import CAPITAL_COST_CATEGORIES, OPERATIONAL_COST_CATEGORIES


_CAPITAL_SCOPE_BY_CATEGORY: dict[str, AnalysisScope] = {
    "server": AnalysisScope.TECHNICAL,
    "client": AnalysisScope.TECHNICAL,
    "network": AnalysisScope.TECHNICAL,
    "licenses": AnalysisScope.SOFTWARE,
}

_OPERATIONAL_SCOPE_BY_CATEGORY: dict[str, AnalysisScope] = {
    "subscription_licenses": AnalysisScope.SOFTWARE,
    "server_rental": AnalysisScope.TECHNICAL,
    "migration": AnalysisScope.IMPLEMENTATION,
    "testing": AnalysisScope.IMPLEMENTATION,
    "backup": AnalysisScope.COMMON,
    "labor_costs": AnalysisScope.COMMON,
    "server_administration": AnalysisScope.TECHNICAL,
}

_FIXED_COMPONENT_TYPE_BY_CATEGORY: dict[str, ComponentType] = {
    "server": ComponentType.SERVER,
    "network": ComponentType.NETWORK_DEVICE,
    "licenses": ComponentType.SOFTWARE_LICENSE,
    "subscription_licenses": ComponentType.SOFTWARE_SUBSCRIPTION,
    "server_rental": ComponentType.SERVER,
    "migration": ComponentType.IMPLEMENTATION_SERVICE,
    "testing": ComponentType.IMPLEMENTATION_SERVICE,
    "backup": ComponentType.BACKUP_SERVICE,
    "labor_costs": ComponentType.SUPPORT_SERVICE,
    "server_administration": ComponentType.SUPPORT_SERVICE,
}


def normalize_runtime_row(row: Mapping[str, Any], *, category: str) -> dict[str, Any]:
    """Return a copy of a runtime row with transition fields filled.

    Existing explicit values always win.  Missing values are inferred from the
    current category contract documented in ``current_subject_schema.md``.  The
    ambiguous ``client`` category is deliberately conservative: it becomes a
    ``workstation`` only when ``client_seats`` is positive or when the row has an
    explicit workstation type.  A client-row without that signal is not counted
    as a workplace automatically.
    """

    category_name = str(category)
    normalized = dict(deepcopy(row))
    normalized.setdefault("category", category_name)

    scope = normalized.get("scope") or infer_scope(category_name)
    if scope is not None:
        normalized["scope"] = str(scope)

    component_type = normalized.get("component_type") or infer_component_type(
        category_name,
        normalized,
    )
    if component_type is not None:
        normalized["component_type"] = str(component_type)

    _normalize_client_seats(normalized, category=category_name)
    return normalized


def normalize_entities(entities: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Normalize every known runtime category and preserve unknown groups."""

    result: dict[str, list[dict[str, Any]]] = {}
    for category, rows in entities.items():
        category_name = str(category)
        if category_name in CAPITAL_COST_CATEGORIES or category_name in OPERATIONAL_COST_CATEGORIES:
            result[category_name] = [normalize_runtime_row(row, category=category_name) for row in rows]
        else:
            result[category_name] = [dict(deepcopy(row)) for row in rows]
    return result


def infer_scope(category: str) -> str | None:
    scope = _CAPITAL_SCOPE_BY_CATEGORY.get(category) or _OPERATIONAL_SCOPE_BY_CATEGORY.get(category)
    return scope.value if scope is not None else None


def infer_component_type(category: str, row: Mapping[str, Any] | None = None) -> str | None:
    if category == "client":
        return _infer_client_component_type(row or {})
    component_type = _FIXED_COMPONENT_TYPE_BY_CATEGORY.get(category)
    return component_type.value if component_type is not None else None


def _infer_client_component_type(row: Mapping[str, Any]) -> str | None:
    explicit = row.get("component_type")
    if explicit:
        return str(explicit)

    if "client_seats" not in row or row.get("client_seats") in {None, ""}:
        return None

    try:
        client_seats = float(row.get("client_seats") or 0.0)
    except (TypeError, ValueError):
        client_seats = 0.0
    if client_seats > 0:
        return ComponentType.WORKSTATION.value
    return ComponentType.PERIPHERAL.value


def _normalize_client_seats(row: dict[str, Any], *, category: str) -> None:
    component_type = str(row.get("component_type") or "")
    if category != "client" and component_type != ComponentType.WORKSTATION.value:
        if category in {"server", "network", "licenses"}:
            row.setdefault("client_seats", 0)
        return

    if "client_seats" in row and row.get("client_seats") not in {None, ""}:
        return

    if component_type == ComponentType.WORKSTATION.value:
        row["client_seats"] = row.get("quantity", 1)
    elif component_type == ComponentType.PERIPHERAL.value:
        row["client_seats"] = 0


class RuntimeEntityNormalizationService:
    """Small class wrapper for dependency-injection friendly use."""

    def normalize_row(self, row: Mapping[str, Any], *, category: str) -> dict[str, Any]:
        return normalize_runtime_row(row, category=category)

    def normalize_entities(
        self,
        entities: Mapping[str, list[Mapping[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        return normalize_entities(entities)

    def infer_scope(self, category: str) -> str | None:
        return infer_scope(category)

    def infer_component_type(self, category: str, row: Mapping[str, Any] | None = None) -> str | None:
        return infer_component_type(category, row)


__all__ = [
    "RuntimeEntityNormalizationService",
    "infer_component_type",
    "infer_scope",
    "normalize_entities",
    "normalize_runtime_row",
]
