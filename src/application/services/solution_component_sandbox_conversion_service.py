"""Manual conversion helpers from legacy sandbox rows to SolutionComponent drafts.

The service is deliberately explicit: it never scans the repository and never
converts rows on startup.  UI code has to pass one selected legacy row and the
user's classification choices.  The returned payload can then be previewed,
saved as a draft, or promoted to strict analytics by the existing runtime and
normalization services.
"""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any, Mapping

from application.ports import EntityRepository
from application.services.solution_component_normalization_service import (
    SolutionComponentNormalizationService,
)
from domain import ComponentType, SolutionComponent, SolutionComponentOrigin
from shared.constants import LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX

_SCOPE_COMPONENT_TYPES: dict[str, tuple[str, ...]] = {
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

_NUMERIC_COMPONENT_FIELDS = {
    "purchase_cost",
    "implementation_cost",
    "migration_cost",
    "testing_cost",
    "monthly_cost",
    "annual_cost",
    "energy_cost",
    "quantity",
}

_TECHNICAL_WORDS = (
    "server",
    "сервер",
    "workstation",
    "рабоч",
    "компьютер",
    "пк",
    "ноутбук",
    "router",
    "роутер",
    "маршрутизатор",
    "switch",
    "коммутатор",
    "network",
    "сеть",
    "мфу",
    "принтер",
    "сканер",
)
_SOFTWARE_WORDS = (
    "software",
    "license",
    "лиценз",
    "подпис",
    "saas",
    "crm",
    "erp",
    "антивирус",
    "офис",
    "1с",
)
_IMPLEMENTATION_WORDS = (
    "implementation",
    "внедрен",
    "миграц",
    "настрой",
    "обучен",
    "тестирован",
    "работ",
    "интеграц",
)
_BACKUP_WORDS = ("backup", "резерв", "бэкап")
_SUPPORT_WORDS = ("support", "поддерж", "сопровожд", "администр")


class SolutionComponentSandboxConversionService:
    """Build and persist explicit conversion drafts for legacy sandbox rows."""

    def __init__(
        self,
        *,
        normalization_service: SolutionComponentNormalizationService | None = None,
    ):
        self.normalization_service = normalization_service or SolutionComponentNormalizationService()

    def build_conversion_draft(
        self,
        row: Mapping[str, Any],
        *,
        sandbox_entity: str,
        row_index: int | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a serializable draft payload for one selected sandbox row.

        ``overrides`` represents the user's choices from the conversion dialog:
        scope, component type, cost split, metrics and the optional strict flag.
        The original sandbox row is preserved in metadata and the sandbox row is
        not mutated or deleted.
        """

        source_row = deepcopy(dict(row))
        hints = self.classify(row, sandbox_entity=sandbox_entity)
        financial = self.suggest_financial_split(source_row, component_type=hints["component_type"])
        article_title = str(source_row.get("_legacy_article_title") or self._title_from_entity(sandbox_entity))
        source_name = str(source_row.get("name") or source_row.get("title") or article_title).strip()
        component_name = source_name or article_title or "Конвертированная sandbox-запись"
        component_id = self._component_id(component_name, sandbox_entity=sandbox_entity, row_index=row_index)
        quantity = self._number(source_row.get("quantity"), default=1.0)

        metadata = {
            "description": str(source_row.get("_legacy_note") or "").strip(),
            "source_category": article_title,
            "legacy_sandbox_origin": {
                "entity_key": sandbox_entity,
                "row_index": row_index,
                "article_title": article_title,
                "expense_type": source_row.get("_legacy_expense_type"),
                "legacy_row_name": source_name,
                "original_row": source_row,
                "archived_after_conversion": False,
                "conversion_mode": "manual",
            },
            "ui_editor": {
                "source_tab": "legacy_infrastructure_sandbox",
                "conversion_action": "manual_convert_to_solution_component",
            },
        }
        metadata = {key: value for key, value in metadata.items() if value is not None and value != ""}

        payload: dict[str, Any] = {
            "id": component_id,
            "name": component_name,
            "scope": hints["scope"],
            "component_type": hints["component_type"],
            "origin": SolutionComponentOrigin.CONVERTED_SANDBOX.value,
            "strict_analysis_participation": False,
            "purchase_cost": financial["purchase_cost"],
            "implementation_cost": financial["implementation_cost"],
            "migration_cost": financial["migration_cost"],
            "testing_cost": financial["testing_cost"],
            "monthly_cost": financial["monthly_cost"],
            "annual_cost": financial["annual_cost"],
            "energy_cost": financial["energy_cost"],
            "quantity": quantity,
            "metrics": self.suggest_metrics(source_row, hints=hints, quantity=quantity),
            "constraints": {},
            "cost_assumptions": self._cost_assumptions(source_row, article_title=article_title),
            "metadata": metadata,
        }
        return self.apply_overrides(payload, overrides or {})

    def classify(self, row: Mapping[str, Any], *, sandbox_entity: str) -> dict[str, str]:
        """Suggest scope and component type while leaving the final choice to UI."""

        text = " ".join(
            str(value)
            for value in (
                sandbox_entity,
                row.get("name"),
                row.get("title"),
                row.get("_legacy_article_title"),
                row.get("_legacy_expense_type"),
            )
            if value
        ).lower()
        if any(word in text for word in _SOFTWARE_WORDS):
            if "подпис" in text or "subscription" in text or "saas" in text:
                return {"scope": "software", "component_type": ComponentType.SOFTWARE_SUBSCRIPTION.value}
            return {"scope": "software", "component_type": ComponentType.SOFTWARE_LICENSE.value}
        if any(word in text for word in _IMPLEMENTATION_WORDS):
            return {"scope": "software", "component_type": ComponentType.IMPLEMENTATION_SERVICE.value}
        if any(word in text for word in _BACKUP_WORDS):
            return {"scope": "technical", "component_type": ComponentType.BACKUP_SERVICE.value}
        if any(word in text for word in _SUPPORT_WORDS):
            return {"scope": "technical", "component_type": ComponentType.SUPPORT_SERVICE.value}
        if "network" in text or "сеть" in text or "router" in text or "switch" in text:
            return {"scope": "technical", "component_type": ComponentType.NETWORK_DEVICE.value}
        if "мфу" in text or "принтер" in text or "сканер" in text:
            return {"scope": "technical", "component_type": ComponentType.PERIPHERAL.value}
        if "workstation" in text or "рабоч" in text or "пк" in text or "ноутбук" in text:
            return {"scope": "technical", "component_type": ComponentType.WORKSTATION.value}
        if any(word in text for word in _TECHNICAL_WORDS):
            return {"scope": "technical", "component_type": ComponentType.SERVER.value}
        if str(row.get("_legacy_expense_type") or "") == "periodic":
            return {"scope": "technical", "component_type": ComponentType.SUPPORT_SERVICE.value}
        return {"scope": "mixed", "component_type": ComponentType.BUNDLE.value}

    def suggest_financial_split(
        self,
        row: Mapping[str, Any],
        *,
        component_type: str,
    ) -> dict[str, float]:
        """Map legacy one-time/monthly cells into explicit financial fields."""

        one_time = self._number(row.get("one_time_cost"), default=0.0)
        monthly = self._number(row.get("monthly_cost"), default=0.0)
        split = {
            "purchase_cost": 0.0,
            "implementation_cost": 0.0,
            "migration_cost": 0.0,
            "testing_cost": 0.0,
            "monthly_cost": monthly,
            "annual_cost": 0.0,
            "energy_cost": 0.0,
        }
        if component_type == ComponentType.IMPLEMENTATION_SERVICE.value:
            split["implementation_cost"] = one_time
        elif component_type == ComponentType.SOFTWARE_SUBSCRIPTION.value:
            split["monthly_cost"] = monthly or one_time / 12.0 if one_time else monthly
        elif component_type in {ComponentType.SUPPORT_SERVICE.value, ComponentType.BACKUP_SERVICE.value}:
            split["implementation_cost"] = one_time
        else:
            split["purchase_cost"] = one_time
        return split

    def suggest_metrics(
        self,
        row: Mapping[str, Any],
        *,
        hints: Mapping[str, str],
        quantity: float,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {"quantity": quantity}
        scope = hints.get("scope")
        component_type = hints.get("component_type")
        if scope == "technical":
            if component_type == ComponentType.WORKSTATION.value:
                metrics.setdefault("client_seats", quantity)
            if component_type == ComponentType.PERIPHERAL.value:
                metrics.setdefault("client_seats", 0.0)
        if scope == "software":
            metrics.setdefault("license_units", quantity)
        return metrics

    def apply_overrides(
        self,
        payload: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = deepcopy(dict(payload))
        extra_metrics = deepcopy(dict(overrides.get("metrics") or {}))
        extra_metadata = deepcopy(dict(overrides.get("metadata") or {}))
        if extra_metrics:
            metrics = deepcopy(dict(result.get("metrics") or {}))
            metrics.update({str(key): value for key, value in extra_metrics.items() if value != ""})
            result["metrics"] = metrics
        if extra_metadata:
            metadata = deepcopy(dict(result.get("metadata") or {}))
            metadata.update(extra_metadata)
            result["metadata"] = metadata
        for key, value in overrides.items():
            if key in {"metrics", "metadata"}:
                continue
            if value is None:
                continue
            if key in _NUMERIC_COMPONENT_FIELDS:
                result[key] = self._number(value, default=0.0)
            elif key == "strict_analysis_participation":
                result[key] = bool(value)
            elif key == "cost_assumptions":
                result[key] = [str(item).strip() for item in value if str(item).strip()]
            elif key == "scope":
                scope = str(value or result.get("scope") or "mixed")
                result[key] = scope
                allowed = _SCOPE_COMPONENT_TYPES.get(scope, _SCOPE_COMPONENT_TYPES["mixed"])
                if str(result.get("component_type")) not in allowed:
                    result["component_type"] = allowed[0]
            elif key == "component_type":
                result[key] = str(value or result.get(key) or ComponentType.BUNDLE.value)
            else:
                result[key] = value
        return result

    def preview_conversion(
        self,
        row: Mapping[str, Any],
        *,
        sandbox_entity: str,
        row_index: int | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> SolutionComponent:
        draft = self.build_conversion_draft(
            row,
            sandbox_entity=sandbox_entity,
            row_index=row_index,
            overrides=overrides,
        )
        return self.normalization_service.normalize(draft)

    def convert_selected_row(
        self,
        repository: EntityRepository,
        *,
        sandbox_entity: str,
        row_index: int,
        overrides: Mapping[str, Any] | None = None,
    ) -> SolutionComponent:
        rows = repository.list(sandbox_entity)
        if row_index < 0 or row_index >= len(rows):
            raise IndexError("legacy sandbox row index is out of range")
        draft = self.build_conversion_draft(
            rows[row_index],
            sandbox_entity=sandbox_entity,
            row_index=row_index,
            overrides=overrides,
        )
        from application.services.solution_component_runtime_service import (  # local import to avoid cycle
            SolutionComponentRuntimeService,
        )

        return SolutionComponentRuntimeService(
            repository,
            normalization_service=self.normalization_service,
        ).add_component(draft)

    def _cost_assumptions(self, row: Mapping[str, Any], *, article_title: str) -> list[str]:
        assumptions = [
            f"Запись вручную преобразована из вспомогательной sandbox-статьи «{article_title}».",
            "Исходная sandbox-запись сохранена и не архивирована автоматически.",
        ]
        note = str(row.get("_legacy_note") or "").strip()
        if note:
            assumptions.append(note)
        return assumptions

    def _component_id(
        self,
        name: str,
        *,
        sandbox_entity: str,
        row_index: int | None,
    ) -> str:
        slug_source = f"{sandbox_entity}:{row_index if row_index is not None else 'row'}:{name}"
        readable = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
        digest = hashlib.sha1(slug_source.encode("utf-8")).hexdigest()[:8]
        if readable:
            return f"converted-{readable}-{digest}"
        return f"converted-sandbox-{digest}"

    def _title_from_entity(self, sandbox_entity: str) -> str:
        if sandbox_entity.startswith(LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX):
            sandbox_entity = sandbox_entity.removeprefix(LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX)
        return sandbox_entity.replace("_", " ").strip() or "Sandbox"

    def _number(self, value: Any, *, default: float = 0.0) -> float:
        try:
            if value is None or str(value).strip() == "":
                return float(default)
            return float(str(value).replace(",", "."))
        except (TypeError, ValueError):
            return float(default)


__all__ = ["SolutionComponentSandboxConversionService"]
