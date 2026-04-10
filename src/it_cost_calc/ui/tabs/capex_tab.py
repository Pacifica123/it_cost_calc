"""Capital costs UI built on reusable widgets and application services."""

from __future__ import annotations

from tkinter import messagebox

from it_cost_calc.application.services.equipment_service import EquipmentService
from it_cost_calc.shared.validation import parse_float, parse_int, require_text
from it_cost_calc.ui.dialogs import RecordFormDialog
from it_cost_calc.ui.tabs.base_scrollable_tab import BaseScrollableTab
from it_cost_calc.ui.widgets import EntityTableSection


class CapexTab(BaseScrollableTab):
    COLUMN_NAMES = {
        "name": "Наименование",
        "quantity": "Количество",
        "price": "Цена",
    }
    ENTITY_CONFIG = {
        "server": ("Серверное оборудование", ("name", "quantity", "price")),
        "client": ("Клиентское оборудование", ("name", "quantity", "price")),
        "network": ("Сетевое оборудование", ("name", "quantity", "price")),
        "licenses": ("Лицензии ПО", ("name", "quantity", "price")),
    }

    def __init__(self, parent, equipment_service: EquipmentService):
        super().__init__(parent)
        self.equipment_service = equipment_service

        self.inner_frame.columnconfigure(0, weight=1)
        self.inner_frame.columnconfigure(1, weight=1)

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
            section.grid(row=index // 2, column=index % 2, padx=10, pady=10, sticky="nsew")
            self.tables[entity_name] = section

        self.refresh_all()

    def _build_dialog(self, title: str, initial_values: dict | None = None) -> dict | None:
        return RecordFormDialog(
            self,
            title=title,
            fields=[
                {"name": "name", "label": "Наименование", "parser": require_text},
                {"name": "quantity", "label": "Количество", "parser": parse_int},
                {"name": "price", "label": "Цена", "parser": parse_float},
            ],
            initial_values=initial_values,
        ).show()

    def refresh_table(self, entity_name: str) -> None:
        self.tables[entity_name].replace_rows(
            self.equipment_service.list_capital_item_rows(entity_name)
        )
        self.update_scrollregion()

    def refresh_all(self) -> None:
        for entity_name in self.tables:
            self.refresh_table(entity_name)

    def add_row(self, entity_name: str) -> None:
        payload = self._build_dialog("Добавить запись")
        if payload is None:
            return
        self.equipment_service.add_capital_item(entity_name, payload)
        self.refresh_table(entity_name)

    def edit_row(self, entity_name: str) -> None:
        selected_index = self.tables[entity_name].get_selected_index()
        if selected_index is None:
            messagebox.showwarning("Нет выбора", "Выберите строку для редактирования", parent=self)
            return
        current_row = self.equipment_service.get_capital_item_row(entity_name, selected_index)
        payload = self._build_dialog("Редактировать запись", initial_values=current_row)
        if payload is None:
            return
        self.equipment_service.update_capital_item(entity_name, selected_index, payload)
        self.refresh_table(entity_name)

    def delete_row(self, entity_name: str) -> None:
        selected_index = self.tables[entity_name].get_selected_index()
        if selected_index is None:
            messagebox.showwarning("Нет выбора", "Выберите строку для удаления", parent=self)
            return
        self.equipment_service.delete_capital_item(entity_name, selected_index)
        self.refresh_table(entity_name)


CapitalCostsTab = CapexTab

__all__ = ["CapexTab", "CapitalCostsTab"]
