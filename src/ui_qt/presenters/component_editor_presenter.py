from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from application.services.solution_component_normalization_service import (
    SolutionComponentNormalizationService,
)
from application.services.solution_component_runtime_service import SolutionComponentRuntimeService
from domain import ComponentType, SolutionComponent
from shared.constants import SOLUTION_COMPONENT_ENTITY
from ui_qt.presenters.app_presenter import QtAppPresenter

_SCOPE_LABELS = {
    "technical": "ТО",
    "software": "ПО",
    "implementation": "Внедрение",
    "mixed": "Смешанный",
}
_SCOPE_HELP = {
    "technical": "Оборудование, сеть, мощность и рабочие места.",
    "software": "Лицензии, пользователи, совместимость и поддержка.",
    "implementation": "Работы, миграция, тестирование и длительность.",
    "mixed": "Смешанные строки лучше позже разложить на отдельные профили.",
}
_COMPONENT_TYPE_LABELS = {
    ComponentType.SERVER.value: "Сервер",
    ComponentType.WORKSTATION.value: "Рабочая станция",
    ComponentType.PERIPHERAL.value: "Периферия",
    ComponentType.NETWORK_DEVICE.value: "Сетевое устройство",
    ComponentType.SOFTWARE_LICENSE.value: "Лицензия",
    ComponentType.SOFTWARE_SUBSCRIPTION.value: "Подписка",
    ComponentType.SOFTWARE_SERVICE.value: "ПО-сервис",
    ComponentType.IMPLEMENTATION_SERVICE.value: "Внедрение",
    ComponentType.SUPPORT_SERVICE.value: "Сопровождение",
    ComponentType.BACKUP_SERVICE.value: "Резервирование",
    ComponentType.BUNDLE.value: "Пакет",
}
_SCOPE_COMPONENT_TYPES = {
    "technical": (
        ComponentType.SERVER.value,
        ComponentType.WORKSTATION.value,
        ComponentType.PERIPHERAL.value,
        ComponentType.NETWORK_DEVICE.value,
        ComponentType.SUPPORT_SERVICE.value,
        ComponentType.BACKUP_SERVICE.value,
    ),
    "software": (
        ComponentType.SOFTWARE_LICENSE.value,
        ComponentType.SOFTWARE_SUBSCRIPTION.value,
        ComponentType.SOFTWARE_SERVICE.value,
        ComponentType.IMPLEMENTATION_SERVICE.value,
        ComponentType.SUPPORT_SERVICE.value,
    ),
    "implementation": (
        ComponentType.IMPLEMENTATION_SERVICE.value,
        ComponentType.SUPPORT_SERVICE.value,
    ),
    "mixed": (
        ComponentType.BUNDLE.value,
        ComponentType.SERVER.value,
        ComponentType.SOFTWARE_SERVICE.value,
        ComponentType.IMPLEMENTATION_SERVICE.value,
        ComponentType.SUPPORT_SERVICE.value,
    ),
}
_PROFILE_METRIC_FIELDS = {
    "technical": (
        ("max_power", "Мощность, Вт"),
        ("client_seats", "Рабочие места"),
        ("performance_score", "Производит."),
        ("reliability_score", "Надёжность"),
        ("hours_per_day", "Часов/день"),
        ("working_days", "Дней/мес."),
    ),
    "software": (
        ("license_units", "Лицензии"),
        ("functionality_score", "Функциональность"),
        ("compatibility_score", "Совместимость"),
        ("support_score", "Поддержка"),
    ),
    "implementation": (
        ("labor_hours", "Трудозатраты"),
        ("migration_complexity", "Миграция"),
        ("testing_coverage", "Тесты"),
        ("duration_days", "Дней"),
    ),
    "mixed": (
        ("bundle_size", "Частей"),
        ("integration_risk_score", "Риск"),
        ("split_priority", "Разбить"),
    ),
}
_NUMERIC_FIELDS = {
    "quantity",
    "purchase_cost",
    "implementation_cost",
    "migration_cost",
    "testing_cost",
    "monthly_cost",
    "annual_cost",
    "energy_cost",
}


@dataclass(frozen=True)
class ComponentOption:
    value: str
    label: str
    help: str = ""


@dataclass(frozen=True)
class MetricField:
    name: str
    label: str


@dataclass(frozen=True)
class ComponentEditorSaveResult:
    index: int
    component: SolutionComponent


class ComponentEditorPresenter:
    """Qt presenter for solution-component CRUD without Tkinter types."""

    def __init__(
        self,
        app_presenter: QtAppPresenter | None = None,
        *,
        normalization_service: SolutionComponentNormalizationService | None = None,
    ) -> None:
        self.app_presenter = app_presenter or QtAppPresenter()
        self.normalization_service = normalization_service or SolutionComponentNormalizationService()
        self.runtime_service = SolutionComponentRuntimeService(
            self.app_presenter.repository,
            normalization_service=self.normalization_service,
        )

    @property
    def entity_name(self) -> str:
        return SOLUTION_COMPONENT_ENTITY

    def scope_options(self) -> list[ComponentOption]:
        return [
            ComponentOption(value=value, label=label, help=_SCOPE_HELP.get(value, ""))
            for value, label in _SCOPE_LABELS.items()
        ]

    def component_type_options(self, scope: str) -> list[ComponentOption]:
        values = _SCOPE_COMPONENT_TYPES.get(scope, _SCOPE_COMPONENT_TYPES["technical"])
        return [
            ComponentOption(value=value, label=_COMPONENT_TYPE_LABELS.get(value, value))
            for value in values
        ]

    def metric_fields(self, scope: str) -> list[MetricField]:
        fields = _PROFILE_METRIC_FIELDS.get(scope, _PROFILE_METRIC_FIELDS["technical"])
        return [MetricField(name=name, label=label) for name, label in fields]

    def list_rows(self) -> list[dict[str, Any]]:
        return self.runtime_service.list_rows()

    def list_table_rows(self) -> list[dict[str, Any]]:
        return [self._component_table_row(component) for component in self.runtime_service.list_components()]

    def status_summary(self) -> str:
        total = len(self.runtime_service.list_components())
        if total == 1:
            return "1 запись"
        return f"{total} записей"

    def new_form_values(self) -> dict[str, Any]:
        return {
            "id": "",
            "name": "",
            "scope": "technical",
            "component_type": ComponentType.SERVER.value,
            "quantity": "1",
            "purchase_cost": "0",
            "implementation_cost": "0",
            "migration_cost": "0",
            "testing_cost": "0",
            "monthly_cost": "0",
            "annual_cost": "0",
            "energy_cost": "0",
            "description": "",
            "strict_analysis_participation": True,
            "metrics": {},
            "cost_assumptions": [],
        }

    def form_values_for_index(self, index: int) -> dict[str, Any]:
        rows = self.runtime_service.list_rows()
        component = SolutionComponent.from_dict(rows[index])
        values = self.new_form_values()
        scope = component.scope.value if component.scope else "technical"
        values.update(
            {
                "id": component.id,
                "name": component.name,
                "scope": scope,
                "component_type": (
                    component.component_type.value
                    if component.component_type
                    else self.component_type_options(scope)[0].value
                ),
                "quantity": _format_number(component.quantity),
                "purchase_cost": _format_number(component.purchase_cost),
                "implementation_cost": _format_number(component.implementation_cost),
                "migration_cost": _format_number(component.migration_cost),
                "testing_cost": _format_number(component.testing_cost),
                "monthly_cost": _format_number(component.monthly_cost),
                "annual_cost": _format_number(component.annual_cost),
                "energy_cost": _format_number(component.energy_cost),
                "description": str(component.metadata.get("description", "")),
                "strict_analysis_participation": bool(component.strict_analysis_participation),
                "metrics": deepcopy(component.metrics),
                "cost_assumptions": list(component.cost_assumptions),
            }
        )
        return values

    def preview(self, values: Mapping[str, Any]) -> str:
        component = self.normalization_service.normalize(build_solution_component_payload(values))
        return format_normalization_preview(component)

    def save(
        self,
        values: Mapping[str, Any],
        *,
        selected_index: int | None = None,
    ) -> ComponentEditorSaveResult:
        payload = build_solution_component_payload(values, require_name=True)
        if selected_index is None:
            existing_count = len(self.runtime_service.list_rows())
            component = self.runtime_service.add_component(payload)
            index = existing_count
        else:
            component = self.runtime_service.update_component(selected_index, payload)
            index = selected_index
        return ComponentEditorSaveResult(index=index, component=component)

    def delete(self, index: int) -> SolutionComponent:
        return self.runtime_service.delete_component(index)

    def _component_table_row(self, component: SolutionComponent) -> dict[str, Any]:
        scope = component.scope.value if component.scope else ""
        component_type = component.component_type.value if component.component_type else ""
        return {
            "name": component.name or component.id or "Без названия",
            "scope": _SCOPE_LABELS.get(scope, scope or "—"),
            "component_type": _COMPONENT_TYPE_LABELS.get(component_type, component_type or "—"),
            "editor_status": "strict" if component.candidate_eligible else "draft",
            "readiness": readiness_label(component),
        }


def build_solution_component_payload(
    form_values: Mapping[str, Any],
    *,
    require_name: bool = False,
) -> dict[str, Any]:
    name = str(form_values.get("name") or "").strip()
    if require_name and not name:
        raise ValueError("Название: пусто")
    component_id = str(form_values.get("id") or "").strip()
    if not component_id and name:
        component_id = generate_component_id(name)

    metrics = {
        str(key): _number_or_text(value)
        for key, value in dict(form_values.get("metrics") or {}).items()
        if str(value).strip() != ""
    }
    quantity = _number_from_value(form_values.get("quantity"), default=1.0, label="Количество")
    if quantity:
        metrics.setdefault("quantity", quantity)

    assumptions_source = form_values.get("cost_assumptions", [])
    if isinstance(assumptions_source, str):
        assumptions = assumptions_source.splitlines()
    else:
        assumptions = [str(item) for item in assumptions_source]
    assumptions = [line.strip() for line in assumptions if line.strip()]

    description = str(form_values.get("description") or "").strip()
    metadata = deepcopy(dict(form_values.get("metadata") or {}))
    if description:
        metadata["description"] = description
    metadata.setdefault("ui_editor", {})
    metadata["ui_editor"].update(
        {
            "source_tab": "qt_component_editor",
            "profile_hint": _SCOPE_HELP.get(str(form_values.get("scope") or ""), ""),
        }
    )

    payload = {
        "id": component_id,
        "name": name,
        "scope": str(form_values.get("scope") or "technical"),
        "component_type": str(form_values.get("component_type") or ComponentType.SERVER.value),
        "origin": "manual",
        "strict_analysis_participation": bool(
            form_values.get("strict_analysis_participation", True)
        ),
        "purchase_cost": _number_from_value(form_values.get("purchase_cost"), label="Покупка"),
        "implementation_cost": _number_from_value(
            form_values.get("implementation_cost"), label="Внедрение"
        ),
        "migration_cost": _number_from_value(form_values.get("migration_cost"), label="Миграция"),
        "testing_cost": _number_from_value(form_values.get("testing_cost"), label="Тесты"),
        "monthly_cost": _number_from_value(form_values.get("monthly_cost"), label="Ежемес."),
        "annual_cost": _number_from_value(form_values.get("annual_cost"), label="Ежегодно"),
        "energy_cost": _number_from_value(form_values.get("energy_cost"), label="Энергия"),
        "quantity": quantity,
        "metrics": metrics,
        "constraints": deepcopy(dict(form_values.get("constraints") or {})),
        "cost_assumptions": assumptions,
        "metadata": metadata,
    }
    if payload["scope"] == "mixed":
        payload["metadata"].setdefault("editor_warnings", []).append(
            "Смешанный компонент лучше разложить на ТО/ПО/внедрение."
        )
    return payload


def format_normalization_preview(component: SolutionComponent | Mapping[str, Any]) -> str:
    normalized = (
        component if isinstance(component, SolutionComponent) else SolutionComponent.from_dict(component)
    )
    warnings = [*normalized.blocking_errors, *normalized.validation_warnings]
    reasons = "\n".join(f"  - {reason}" for reason in warnings)
    if not reasons:
        reasons = "  - блокирующие причины не обнаружены"
    return (
        f"Состояние: {normalized.normalization_state.value}\n"
        f"CandidateConfiguration: {'да' if normalized.candidate_eligible else 'нет'}\n"
        f"TCO: {'да' if normalized.tco_eligible else 'нет'}\n"
        f"Энергия: {'да' if normalized.energy_eligible else 'нет'}\n"
        f"NPV: {'да' if normalized.npv_eligible else 'нет'}\n"
        f"Причины:\n{reasons}"
    )


def readiness_label(component: SolutionComponent) -> str:
    if component.analysis_ready:
        return "готов"
    if component.candidate_eligible:
        return "нужна TCO"
    if component.blocking_errors:
        return "заблокирован"
    return "черновик"


def generate_component_id(name: str) -> str:
    text = str(name).strip().lower()
    slug = re.sub(r"[^a-z0-9а-яё]+", "-", text).strip("-")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    if slug:
        return f"component-{slug}-{digest}"
    return f"component-{digest}"


def _number_from_value(value: Any, *, default: float = 0.0, label: str = "Число") -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    raw = str(value).replace(" ", "").replace(",", ".").strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{label}: нужно число") from exc


def _number_or_text(value: Any) -> Any:
    if value is None:
        return ""
    raw = str(value).strip()
    if raw == "":
        return ""
    try:
        return float(raw.replace(" ", "").replace(",", "."))
    except ValueError:
        return raw


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value or "")
    if number.is_integer():
        return str(int(number))
    return str(number)


__all__ = [
    "ComponentEditorPresenter",
    "ComponentEditorSaveResult",
    "ComponentOption",
    "MetricField",
    "build_solution_component_payload",
    "format_normalization_preview",
    "generate_component_id",
    "readiness_label",
]
