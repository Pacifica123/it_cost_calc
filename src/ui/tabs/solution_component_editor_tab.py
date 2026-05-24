"""MVP editor tab for user-defined solution components.

The tab is intentionally small: it lets a user create a structured
``SolutionComponent`` row, preview normalization before saving, and keep
incomplete rows as drafts.  Strict analytics are still controlled by application
services, not by widget code.
"""

from __future__ import annotations

import hashlib
import logging
import re
import tkinter as tk
from copy import deepcopy
from tkinter import messagebox, ttk
from typing import Any, Mapping

from application.services.solution_component_normalization_service import (
    SolutionComponentNormalizationService,
)
from application.services.solution_component_runtime_service import SolutionComponentRuntimeService
from domain import ComponentType, SolutionComponent
from shared.constants import SOLUTION_COMPONENT_ENTITY
from shared.validation import require_text
from ui.tabs.base_scrollable_tab import BaseScrollableTab

logger = logging.getLogger(__name__)

_SCOPE_LABELS = {
    "technical": "ТО: оборудование, сеть, мощность, рабочие места",
    "software": "ПО: лицензии, пользователи, совместимость, поддержка",
    "implementation": "Внедрение: работы, миграция, тестирование, длительность",
    "mixed": "Смешанный компонент: требует проверки и возможного разбиения",
}

_COMPONENT_TYPE_LABELS = {
    ComponentType.SERVER.value: "Сервер",
    ComponentType.WORKSTATION.value: "Рабочая станция",
    ComponentType.PERIPHERAL.value: "Периферия",
    ComponentType.NETWORK_DEVICE.value: "Сетевое устройство",
    ComponentType.SOFTWARE_LICENSE.value: "Программная лицензия",
    ComponentType.SOFTWARE_SUBSCRIPTION.value: "Подписка на ПО",
    ComponentType.SOFTWARE_SERVICE.value: "Программный сервис",
    ComponentType.IMPLEMENTATION_SERVICE.value: "Работы по внедрению",
    ComponentType.SUPPORT_SERVICE.value: "Сопровождение",
    ComponentType.BACKUP_SERVICE.value: "Резервное копирование",
    ComponentType.BUNDLE.value: "Пакет / смешанный набор",
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
        ("performance_score", "Производительность, балл"),
        ("reliability_score", "Надёжность, балл"),
        ("hours_per_day", "Часы работы в день"),
        ("working_days", "Рабочие дни в месяце"),
    ),
    "software": (
        ("license_units", "Лицензии / пользователи"),
        ("functionality_score", "Функциональность, балл"),
        ("compatibility_score", "Совместимость, балл"),
        ("support_score", "Поддержка, балл"),
    ),
    "implementation": (
        ("labor_hours", "Трудозатраты, ч"),
        ("migration_complexity", "Сложность миграции, балл"),
        ("testing_coverage", "Покрытие тестированием, балл"),
        ("duration_days", "Длительность, дни"),
    ),
    "mixed": (
        ("bundle_size", "Количество частей в пакете"),
        ("integration_risk_score", "Риск интеграции, балл"),
        ("split_priority", "Приоритет разбиения, балл"),
    ),
}

_NUMERIC_FORM_FIELDS = {
    "purchase_cost",
    "implementation_cost",
    "migration_cost",
    "testing_cost",
    "monthly_cost",
    "annual_cost",
    "energy_cost",
    "quantity",
}


class SolutionComponentEditorTab(BaseScrollableTab):
    """Small structured editor over the C3 ``solution_components`` runtime section."""

    TABLE_COLUMNS = (
        "name",
        "scope",
        "component_type",
        "editor_status",
        "readiness",
    )
    TABLE_HEADINGS = {
        "name": "Компонент",
        "scope": "Профиль",
        "component_type": "Тип",
        "editor_status": "Режим",
        "readiness": "Готовность",
    }

    def __init__(
        self,
        parent,
        runtime_service: SolutionComponentRuntimeService,
        *,
        normalization_service: SolutionComponentNormalizationService | None = None,
    ):
        super().__init__(parent)
        self.runtime_service = runtime_service
        self.normalization_service = normalization_service or runtime_service.normalization_service
        self.selected_index: int | None = None
        self.common_vars: dict[str, tk.Variable] = {}
        self.metric_vars: dict[str, tk.StringVar] = {}
        self.preview_var = tk.StringVar(value="Нажмите «Проверить нормализацию», чтобы увидеть статус до сохранения.")
        self.status_var = tk.StringVar(value="Компоненты решения ещё не обновлялись в этой сессии.")

        self.inner_frame.columnconfigure(0, weight=1)
        self._build_intro()
        self._build_form()
        self._build_preview()
        self._build_table()
        self.refresh_components()
        self.update_scrollregion()

    def _build_intro(self) -> None:
        intro = tk.Label(
            self.inner_frame,
            text=(
                "Редактор компонентов — advanced-режим для структурированного описания частей "
                "ИТ-решения. Здесь компонент отличается от свободной заметки: у него есть профиль, "
                "тип, стоимость, метрики и статус готовности к аналитике."
            ),
            anchor="w",
            justify="left",
            wraplength=1080,
        )
        intro.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))

        hint = tk.Label(
            self.inner_frame,
            text=(
                f"Данные сохраняются в runtime-секции {SOLUTION_COMPONENT_ENTITY!r}. "
                "Неполные строки можно оставить черновиками: они будут видны в отчёте, но не попадут "
                "в строгую аналитику до нормализации. Старая ИТ-песочница остаётся архивом рядом."
            ),
            anchor="w",
            justify="left",
            fg="#555555",
            wraplength=1080,
        )
        hint.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

    def _build_form(self) -> None:
        self.form_frame = tk.LabelFrame(self.inner_frame, text="Новый / редактируемый компонент", padx=10, pady=8)
        self.form_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))
        self.form_frame.columnconfigure(1, weight=1)
        self.form_frame.columnconfigure(3, weight=1)

        self.common_vars = {
            "id": tk.StringVar(),
            "name": tk.StringVar(),
            "scope": tk.StringVar(value="technical"),
            "component_type": tk.StringVar(value=ComponentType.SERVER.value),
            "quantity": tk.StringVar(value="1"),
            "purchase_cost": tk.StringVar(value="0"),
            "implementation_cost": tk.StringVar(value="0"),
            "migration_cost": tk.StringVar(value="0"),
            "testing_cost": tk.StringVar(value="0"),
            "monthly_cost": tk.StringVar(value="0"),
            "annual_cost": tk.StringVar(value="0"),
            "energy_cost": tk.StringVar(value="0"),
            "description": tk.StringVar(),
            "strict_analysis_participation": tk.BooleanVar(value=True),
        }

        self._add_entry("id", "ID компонента", row=0, column=0)
        self._add_entry("name", "Название компонента", row=1, column=0)
        self._add_entry("description", "Назначение / описание", row=2, column=0, columnspan=3)

        tk.Label(self.form_frame, text="Профиль").grid(row=0, column=2, sticky="w", padx=(12, 4), pady=4)
        self.scope_combo = ttk.Combobox(
            self.form_frame,
            textvariable=self.common_vars["scope"],
            values=tuple(_SCOPE_LABELS.keys()),
            state="readonly",
            width=26,
        )
        self.scope_combo.grid(row=0, column=3, sticky="ew", pady=4)
        self.scope_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_scope_change())

        tk.Label(self.form_frame, text="Тип компонента").grid(row=1, column=2, sticky="w", padx=(12, 4), pady=4)
        self.type_combo = ttk.Combobox(
            self.form_frame,
            textvariable=self.common_vars["component_type"],
            values=_SCOPE_COMPONENT_TYPES["technical"],
            state="readonly",
            width=26,
        )
        self.type_combo.grid(row=1, column=3, sticky="ew", pady=4)

        costs = tk.LabelFrame(self.form_frame, text="Стоимость", padx=8, pady=6)
        costs.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        costs.columnconfigure(1, weight=1)
        costs.columnconfigure(3, weight=1)
        self._add_entry("quantity", "Количество", row=0, column=0, parent=costs)
        self._add_entry("purchase_cost", "Покупка", row=0, column=2, parent=costs)
        self._add_entry("implementation_cost", "Внедрение", row=1, column=0, parent=costs)
        self._add_entry("migration_cost", "Миграция", row=1, column=2, parent=costs)
        self._add_entry("testing_cost", "Тестирование", row=2, column=0, parent=costs)
        self._add_entry("monthly_cost", "Ежемесячно", row=2, column=2, parent=costs)
        self._add_entry("annual_cost", "Ежегодно", row=3, column=0, parent=costs)
        self._add_entry("energy_cost", "Энергия / мес.", row=3, column=2, parent=costs)

        self.profile_frame = tk.LabelFrame(self.form_frame, text="Профильные метрики", padx=8, pady=6)
        self.profile_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        self.profile_frame.columnconfigure(1, weight=1)
        self.profile_frame.columnconfigure(3, weight=1)

        assumptions_frame = tk.LabelFrame(self.form_frame, text="Допущения и комментарии", padx=8, pady=6)
        assumptions_frame.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        assumptions_frame.columnconfigure(0, weight=1)
        self.assumptions_text = tk.Text(assumptions_frame, height=4, wrap="word")
        self.assumptions_text.grid(row=0, column=0, sticky="ew")

        options_frame = tk.Frame(self.form_frame)
        options_frame.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        tk.Checkbutton(
            options_frame,
            text="Участвует в строгой аналитике после нормализации",
            variable=self.common_vars["strict_analysis_participation"],
        ).pack(side="left")

        buttons = tk.Frame(self.form_frame)
        buttons.grid(row=7, column=0, columnspan=4, sticky="e", pady=(8, 0))
        tk.Button(buttons, text="Проверить нормализацию", command=self.preview_current_form).pack(side="left", padx=(0, 6))
        tk.Button(buttons, text="Сохранить", command=self.save_current_form).pack(side="left", padx=(0, 6))
        tk.Button(buttons, text="Очистить форму", command=self.clear_form).pack(side="left")

        self._rebuild_profile_fields()

    def _add_entry(
        self,
        key: str,
        label: str,
        *,
        row: int,
        column: int,
        parent: tk.Widget | None = None,
        columnspan: int = 1,
    ) -> tk.Entry:
        parent = parent or self.form_frame
        tk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 4), pady=3)
        entry = tk.Entry(parent, textvariable=self.common_vars[key], width=28)
        entry.grid(row=row, column=column + 1, sticky="ew", pady=3, columnspan=columnspan)
        return entry

    def _build_preview(self) -> None:
        frame = tk.LabelFrame(self.inner_frame, text="Предпросмотр нормализации до сохранения", padx=10, pady=8)
        frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 8))
        frame.columnconfigure(0, weight=1)
        self.preview_label = tk.Label(
            frame,
            textvariable=self.preview_var,
            anchor="w",
            justify="left",
            wraplength=1080,
        )
        self.preview_label.grid(row=0, column=0, sticky="ew")

    def _build_table(self) -> None:
        frame = tk.LabelFrame(self.inner_frame, text="Сохранённые компоненты", padx=10, pady=8)
        frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))
        frame.columnconfigure(0, weight=1)

        self.components_table = ttk.Treeview(
            frame,
            columns=self.TABLE_COLUMNS,
            show="headings",
            height=7,
        )
        for column in self.TABLE_COLUMNS:
            self.components_table.heading(column, text=self.TABLE_HEADINGS[column])
            self.components_table.column(column, stretch=tk.YES, width=160)
        self.components_table.grid(row=0, column=0, sticky="ew")

        buttons = tk.Frame(frame)
        buttons.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        tk.Button(buttons, text="Загрузить в форму", command=self.load_selected_component).pack(side="left")
        tk.Button(buttons, text="Удалить", command=self.delete_selected_component).pack(side="left", padx=(6, 0))
        tk.Button(buttons, text="Обновить список", command=self.refresh_components).pack(side="left", padx=(6, 0))
        tk.Label(buttons, textvariable=self.status_var, fg="#555555").pack(side="left", padx=(12, 0))

    def _on_scope_change(self) -> None:
        scope = self.common_vars["scope"].get() or "technical"
        allowed_types = _SCOPE_COMPONENT_TYPES.get(scope, _SCOPE_COMPONENT_TYPES["technical"])
        self.type_combo.configure(values=allowed_types)
        if self.common_vars["component_type"].get() not in allowed_types:
            self.common_vars["component_type"].set(allowed_types[0])
        self._rebuild_profile_fields()
        self.preview_current_form(show_errors=False)

    def _rebuild_profile_fields(self) -> None:
        for child in self.profile_frame.winfo_children():
            child.destroy()
        self.metric_vars = {}
        scope = self.common_vars["scope"].get() or "technical"
        fields = _PROFILE_METRIC_FIELDS.get(scope, _PROFILE_METRIC_FIELDS["technical"])
        tk.Label(
            self.profile_frame,
            text=_SCOPE_LABELS.get(scope, scope),
            anchor="w",
            justify="left",
            fg="#555555",
        ).grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 4))
        for index, (name, label) in enumerate(fields, start=1):
            row = (index + 1) // 2
            column = 0 if index % 2 else 2
            var = tk.StringVar(value="")
            self.metric_vars[name] = var
            tk.Label(self.profile_frame, text=label).grid(row=row, column=column, sticky="w", padx=(0, 4), pady=3)
            tk.Entry(self.profile_frame, textvariable=var, width=24).grid(row=row, column=column + 1, sticky="ew", pady=3)
        if scope == "mixed":
            tk.Label(
                self.profile_frame,
                text="Смешанный компонент лучше сохранить как черновик и позже разложить на ТО/ПО/внедрение.",
                anchor="w",
                justify="left",
                fg="#7a5500",
                wraplength=980,
            ).grid(row=4, column=0, columnspan=4, sticky="ew", pady=(6, 0))

    def preview_current_form(self, *, show_errors: bool = True) -> None:
        try:
            component = self._component_from_form(require_name=False)
        except ValueError as error:
            if show_errors:
                messagebox.showerror("Ошибка ввода", str(error), parent=self)
            self.preview_var.set(f"Не удалось собрать компонент: {error}")
            return
        normalized = self.normalization_service.normalize(component)
        self.preview_var.set(format_normalization_preview(normalized))

    def save_current_form(self) -> None:
        try:
            component = self._component_from_form(require_name=True)
        except ValueError as error:
            messagebox.showerror("Ошибка ввода", str(error), parent=self)
            return

        normalized = self.normalization_service.normalize(component)
        if self.selected_index is None:
            normalized = self.runtime_service.add_component(normalized)
            logger.info("Создан SolutionComponent через UI: %s", normalized.id)
        else:
            normalized = self.runtime_service.update_component(self.selected_index, normalized)
            logger.info("Обновлён SolutionComponent через UI: %s", normalized.id)
        self.preview_var.set(format_normalization_preview(normalized))
        self.refresh_components()
        self.clear_form(keep_preview=True)

    def load_selected_component(self) -> None:
        index = self._selected_index()
        if index is None:
            messagebox.showwarning("Нет выбора", "Выберите компонент для загрузки в форму", parent=self)
            return
        rows = self.runtime_service.list_rows()
        if index >= len(rows):
            messagebox.showwarning("Нет строки", "Сохранённая строка уже недоступна. Обновите список.", parent=self)
            return
        self.selected_index = index
        self._fill_form(rows[index])
        self.preview_current_form(show_errors=False)

    def delete_selected_component(self) -> None:
        index = self._selected_index()
        if index is None:
            messagebox.showwarning("Нет выбора", "Выберите компонент для удаления", parent=self)
            return
        if not messagebox.askyesno("Удалить компонент?", "Удалить выбранный компонент решения?", parent=self):
            return
        deleted = self.runtime_service.delete_component(index)
        logger.info("Удалён SolutionComponent через UI: %s", deleted.id)
        self.refresh_components()
        self.clear_form()

    def refresh_components(self) -> None:
        for item in self.components_table.get_children():
            self.components_table.delete(item)
        components = self.runtime_service.list_components()
        for component in components:
            self.components_table.insert(
                "",
                "end",
                values=(
                    component.name or component.id or "Без названия",
                    component.scope.value if component.scope else "—",
                    component.component_type.value if component.component_type else "—",
                    "strict" if component.candidate_eligible else "draft",
                    readiness_label(component),
                ),
            )
        self.status_var.set(
            f"Всего: {len(components)}; строгих: {sum(1 for item in components if item.candidate_eligible)}; "
            f"черновиков/исключённых: {sum(1 for item in components if not item.candidate_eligible)}"
        )
        self.update_scrollregion()

    def clear_form(self, *, keep_preview: bool = False) -> None:
        self.selected_index = None
        for key, variable in self.common_vars.items():
            if isinstance(variable, tk.BooleanVar):
                variable.set(True)
            elif key == "scope":
                variable.set("technical")
            elif key == "component_type":
                variable.set(ComponentType.SERVER.value)
            elif key == "quantity":
                variable.set("1")
            elif key in _NUMERIC_FORM_FIELDS:
                variable.set("0")
            else:
                variable.set("")
        self.assumptions_text.delete("1.0", "end")
        self.type_combo.configure(values=_SCOPE_COMPONENT_TYPES["technical"])
        self._rebuild_profile_fields()
        if not keep_preview:
            self.preview_var.set("Нажмите «Проверить нормализацию», чтобы увидеть статус до сохранения.")

    def _fill_form(self, row: Mapping[str, Any]) -> None:
        component = SolutionComponent.from_dict(row)
        self.common_vars["id"].set(component.id)
        self.common_vars["name"].set(component.name)
        self.common_vars["scope"].set(component.scope.value if component.scope else "technical")
        self._on_scope_change()
        self.common_vars["component_type"].set(
            component.component_type.value if component.component_type else ComponentType.SERVER.value
        )
        for key in _NUMERIC_FORM_FIELDS:
            self.common_vars[key].set(_format_number(getattr(component, key)))
        self.common_vars["description"].set(str(component.metadata.get("description", "")))
        self.common_vars["strict_analysis_participation"].set(
            bool(component.strict_analysis_participation)
        )
        self.assumptions_text.delete("1.0", "end")
        self.assumptions_text.insert("1.0", "\n".join(component.cost_assumptions))
        for name, var in self.metric_vars.items():
            if name in component.metrics:
                var.set(_format_number(component.metrics[name]))

    def _component_from_form(self, *, require_name: bool) -> dict[str, Any]:
        raw_name = str(self.common_vars["name"].get()).strip()
        name = require_text(raw_name, "Название компонента") if require_name else raw_name
        component_id = str(self.common_vars["id"].get()).strip()
        if not component_id and name:
            component_id = self._unique_component_id(generate_component_id(name))
            self.common_vars["id"].set(component_id)

        form_values = {
            key: variable.get()
            for key, variable in self.common_vars.items()
            if key != "strict_analysis_participation"
        }
        form_values["id"] = component_id
        form_values["name"] = name
        form_values["strict_analysis_participation"] = bool(
            self.common_vars["strict_analysis_participation"].get()
        )
        form_values["metrics"] = {name: var.get() for name, var in self.metric_vars.items()}
        form_values["cost_assumptions"] = self.assumptions_text.get("1.0", "end").splitlines()
        return build_solution_component_payload(form_values)

    def _unique_component_id(self, base: str) -> str:
        existing = {
            str(row.get("id") or "")
            for index, row in enumerate(self.runtime_service.list_rows())
            if index != self.selected_index
        }
        candidate = base
        counter = 2
        while candidate in existing:
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    def _selected_index(self) -> int | None:
        selected_item = self.components_table.focus()
        if not selected_item:
            return None
        children = list(self.components_table.get_children())
        if selected_item not in children:
            return None
        return children.index(selected_item)


def build_solution_component_payload(form_values: Mapping[str, Any]) -> dict[str, Any]:
    """Build a serializable SolutionComponent payload from UI-like values.

    The helper is GUI-free on purpose, so smoke scripts can validate the C4 form
    contract without a Tk display and without using pytest.
    """

    name = str(form_values.get("name") or "").strip()
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

    assumptions = [str(line).strip() for line in form_values.get("cost_assumptions", [])]
    assumptions = [line for line in assumptions if line]
    description = str(form_values.get("description") or "").strip()
    metadata = deepcopy(dict(form_values.get("metadata") or {}))
    if description:
        metadata["description"] = description
    metadata.setdefault("ui_editor", {})
    metadata["ui_editor"].update(
        {
            "source_tab": "solution_component_editor",
            "profile_hint": _SCOPE_LABELS.get(str(form_values.get("scope") or ""), ""),
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
        "testing_cost": _number_from_value(form_values.get("testing_cost"), label="Тестирование"),
        "monthly_cost": _number_from_value(form_values.get("monthly_cost"), label="Ежемесячно"),
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
            "Смешанный компонент может потребовать разбиения на отдельные ТО/ПО/внедрение."
        )
    return payload


def format_normalization_preview(component: SolutionComponent | Mapping[str, Any]) -> str:
    normalized = component if isinstance(component, SolutionComponent) else SolutionComponent.from_dict(component)
    warnings = [*normalized.blocking_errors, *normalized.validation_warnings]
    if warnings:
        reasons = "\n".join(f"  - {reason}" for reason in warnings)
    else:
        reasons = "  - обязательные блокирующие причины не обнаружены"
    return (
        f"Состояние: {normalized.normalization_state.value}\n"
        f"CandidateConfiguration: {'да' if normalized.candidate_eligible else 'нет'}\n"
        f"TCO: {'да' if normalized.tco_eligible else 'нет'}\n"
        f"Энергетический расчёт: {'да' if normalized.energy_eligible else 'нет'}\n"
        f"NPV: {'да' if normalized.npv_eligible else 'нет'}\n"
        f"Причины/предупреждения:\n{reasons}"
    )


def readiness_label(component: SolutionComponent) -> str:
    if component.analysis_ready:
        return "готов к аналитике"
    if component.candidate_eligible:
        return "нужна стоимость для TCO/NPV"
    if component.blocking_errors:
        return "заблокирован"
    return "черновик"


def generate_component_id(name: str) -> str:
    text = str(name).strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    if slug:
        return f"component-{slug}-{digest}"
    return f"component-{digest}"


def _number_from_value(value: Any, *, default: float = 0.0, label: str = "Число") -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    raw = str(value).replace(",", ".").strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{label}: ожидается число, получено {value!r}") from exc


def _number_or_text(value: Any) -> Any:
    if value is None:
        return ""
    raw = str(value).strip()
    if raw == "":
        return ""
    try:
        return float(raw.replace(",", "."))
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
    "SolutionComponentEditorTab",
    "build_solution_component_payload",
    "format_normalization_preview",
    "generate_component_id",
    "readiness_label",
]
