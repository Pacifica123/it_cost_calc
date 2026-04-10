"""Operational costs UI built on reusable widgets and application services."""

from __future__ import annotations

from tkinter import messagebox

from it_cost_calc.application.services.equipment_service import EquipmentService
from it_cost_calc.shared.validation import parse_float, require_text
from it_cost_calc.ui.dialogs import RecordFormDialog
from it_cost_calc.ui.tabs.base_scrollable_tab import BaseScrollableTab
from it_cost_calc.ui.widgets import EntityTableSection


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

    def __init__(self, parent, equipment_service: EquipmentService):
        super().__init__(parent)
        self.equipment_service = equipment_service

        for column in range(3):
            self.inner_frame.columnconfigure(column, weight=1)

        self.tables: dict[str, EntityTableSection] = {}
        for index, (entity_name, (title, columns)) in enumerate(self.ENTITY_CONFIG.items()):
            section = EntityTableSection(
                self.inner_frame,
                title=title,
                columns=columns,
                column_names=self.COLUMN_NAMES,
                on_add=lambda e=entity_name: self.add_row(e),
                on_edit=lambda e=entity_name: self.edit_row(e),
                on_delete=lambda e=entity_name: self.delete_row(e),
            )
            section.grid(row=index // 3, column=index % 3, padx=10, pady=10, sticky="nsew")
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
