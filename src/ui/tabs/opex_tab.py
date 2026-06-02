"""Operational costs UI built on reusable widgets and application services."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from typing import Mapping

from application.services.equipment_service import EquipmentService
from shared.validation import parse_float, require_text
from ui.dialogs import RecordFormDialog
from ui.tabs.base_scrollable_tab import BaseScrollableTab
from ui.widgets import EntityTableSection
from ui.theme import PANEL_BODY


OperationalEntityConfig = Mapping[str, tuple[str, tuple[str, ...]]]


class OpexTab(BaseScrollableTab):
    COLUMN_NAMES = {
        "name": "Наименование",
        "monthly_cost": "Ежемесячная стоимость",
        "one_time_cost": "Единовременная стоимость",
    }
    ENTITY_CONFIG = {
        "subscription_licenses": ("Лицензии по подписке", ("name", "monthly_cost")),
        "server_rental": ("Аренда серверов", ("name", "monthly_cost")),
        "migration": ("Миграция", ("name", "one_time_cost")),
        "testing": ("Тестирование", ("name", "one_time_cost")),
        "backup": ("Резервирование", ("name", "monthly_cost")),
        "labor_costs": ("Оплата труда", ("name", "monthly_cost")),
        "server_administration": ("Администрирование серверов", ("name", "monthly_cost")),
    }

    def __init__(
        self,
        parent,
        equipment_service: EquipmentService,
        *,
        entity_config: OperationalEntityConfig | None = None,
        intro_text: str | None = None,
        columns_count: int = 3,
        compact_tables: bool = False,
    ):
        super().__init__(parent)
        self.equipment_service = equipment_service
        self.entity_config = dict(entity_config or self.ENTITY_CONFIG)
        self.columns_count = max(1, int(columns_count))
        self.compact_tables = compact_tables

        for column in range(self.columns_count):
            self.inner_frame.columnconfigure(column, weight=1)

        row_offset = 0
        if intro_text:
            label = tk.Label(
                self.inner_frame,
                text=intro_text,
                anchor="w",
                justify="left",
                wraplength=880,
                background=PANEL_BODY,
            )
            label.grid(
                row=0,
                column=0,
                columnspan=self.columns_count,
                padx=10,
                pady=(10, 0),
                sticky="ew",
            )
            label.bind(
                "<Configure>",
                lambda event, widget=label: widget.configure(wraplength=max(260, event.width - 20)),
            )
            row_offset = 1

        self.tables: dict[str, EntityTableSection] = {}
        for index, (entity_name, (title, columns)) in enumerate(self.entity_config.items()):
            section = EntityTableSection(
                self.inner_frame,
                title=title,
                columns=columns,
                column_names=self.COLUMN_NAMES,
                on_add=lambda e=entity_name: self.add_row(e),
                on_edit=lambda e=entity_name: self.edit_row(e),
                on_delete=lambda e=entity_name: self.delete_row(e),
                compact=self.compact_tables,
            )
            section.grid(
                row=row_offset + index // self.columns_count,
                column=index % self.columns_count,
                padx=10,
                pady=10,
                sticky="nsew",
            )
            self.tables[entity_name] = section

        self.refresh_all()

    def _build_dialog(
        self, entity_name: str, title: str, initial_values: dict | None = None
    ) -> dict | None:
        cost_field_name = (
            "one_time_cost" if entity_name in {"migration", "testing"} else "monthly_cost"
        )
        cost_label = self.COLUMN_NAMES[cost_field_name]
        return RecordFormDialog(
            self,
            title=title,
            fields=[
                {"name": "name", "label": "Наименование", "parser": require_text},
                {"name": cost_field_name, "label": cost_label, "parser": parse_float},
            ],
            initial_values=initial_values,
        ).show()

    def refresh_table(self, entity_name: str) -> None:
        self.tables[entity_name].replace_rows(
            self.equipment_service.list_operational_item_rows(entity_name)
        )
        self.update_scrollregion()

    def refresh_all(self) -> None:
        for entity_name in self.tables:
            self.refresh_table(entity_name)

    def add_row(self, entity_name: str) -> None:
        payload = self._build_dialog(entity_name, "Добавить запись")
        if payload is None:
            return
        self.equipment_service.add_operational_item(entity_name, payload)
        self.refresh_table(entity_name)

    def edit_row(self, entity_name: str) -> None:
        selected_index = self.tables[entity_name].get_selected_index()
        if selected_index is None:
            messagebox.showwarning("Нет выбора", "Выберите строку для редактирования", parent=self)
            return
        current_row = self.equipment_service.get_operational_item_row(entity_name, selected_index)
        payload = self._build_dialog(
            entity_name, "Редактировать запись", initial_values=current_row
        )
        if payload is None:
            return
        self.equipment_service.update_operational_item(entity_name, selected_index, payload)
        self.refresh_table(entity_name)

    def delete_row(self, entity_name: str) -> None:
        selected_index = self.tables[entity_name].get_selected_index()
        if selected_index is None:
            messagebox.showwarning("Нет выбора", "Выберите строку для удаления", parent=self)
            return
        self.equipment_service.delete_operational_item(entity_name, selected_index)
        self.refresh_table(entity_name)


OperationalCostsTab = OpexTab

__all__ = ["OpexTab", "OperationalCostsTab"]
