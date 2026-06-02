"""Capital costs UI built on reusable widgets and application services."""

from __future__ import annotations

from tkinter import messagebox
from typing import ClassVar, Mapping

from application.services.equipment_service import EquipmentService
from shared.validation import parse_float, parse_int, require_text
from ui.dialogs import RecordFormDialog
from ui.tabs.base_scrollable_tab import BaseScrollableTab
from ui.widgets import EntityTableSection
from ui.theme import PANEL_BODY


EntityConfig = Mapping[str, tuple[str, tuple[str, ...]]]


class CapexTab(BaseScrollableTab):
    COLUMN_NAMES: ClassVar[dict[str, str]] = {
        "name": "Наименование",
        "quantity": "Количество",
        "price": "Цена",
    }
    ENTITY_CONFIG: ClassVar[dict[str, tuple[str, tuple[str, ...]]]] = {
        "server": ("Серверное оборудование", ("name", "quantity", "price")),
        "client": ("Клиентское оборудование", ("name", "quantity", "price")),
        "network": ("Сетевое оборудование", ("name", "quantity", "price")),
        "licenses": ("Лицензии ПО", ("name", "quantity", "price")),
    }

    def __init__(
        self,
        parent,
        equipment_service: EquipmentService,
        *,
        entity_config: EntityConfig | None = None,
        intro_text: str | None = None,
        columns_count: int = 2,
        compact_tables: bool = False,
    ):
        super().__init__(parent)
        self.equipment_service = equipment_service
        self.entity_config = dict(entity_config or self.ENTITY_CONFIG)
        self.columns_count = max(1, int(columns_count))
        self.compact_tables = compact_tables

        for column in range(self.columns_count):
            self.inner_frame.columnconfigure(column, weight=1, uniform="capex-columns")

        if intro_text:
            import tkinter as tk

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
        else:
            row_offset = 0

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


class TechnicalEquipmentTab(CapexTab):
    ENTITY_CONFIG: ClassVar[dict[str, tuple[str, tuple[str, ...]]]] = {
        "server": ("Серверное оборудование", ("name", "quantity", "price")),
        "client": ("Клиентское оборудование", ("name", "quantity", "price")),
        "network": ("Сетевое оборудование", ("name", "quantity", "price")),
    }

    def __init__(
        self,
        parent,
        equipment_service: EquipmentService,
        *,
        columns_count: int = 2,
        compact_tables: bool = False,
    ):
        super().__init__(
            parent,
            equipment_service,
            entity_config=self.ENTITY_CONFIG,
            columns_count=columns_count,
            compact_tables=compact_tables,
            intro_text=(
                "ТО — техническое обеспечение: серверы, клиентские устройства и сеть. "
                "Именно эта область участвует в расчётах мощности и клиентских мест."
            ),
        )


class SoftwareTab(CapexTab):
    ENTITY_CONFIG: ClassVar[dict[str, tuple[str, tuple[str, ...]]]] = {
        "licenses": ("Лицензии ПО", ("name", "quantity", "price")),
    }

    def __init__(
        self,
        parent,
        equipment_service: EquipmentService,
        *,
        columns_count: int = 1,
        compact_tables: bool = False,
    ):
        super().__init__(
            parent,
            equipment_service,
            entity_config=self.ENTITY_CONFIG,
            columns_count=columns_count,
            compact_tables=compact_tables,
            intro_text=(
                "ПО — программное обеспечение: лицензии и программные позиции. "
                "Для этой области анализы используют стоимость и количество лицензий, "
                "без расчёта мощности и клиентских ПК."
            ),
        )


CapitalCostsTab = CapexTab

__all__ = ["CapexTab", "CapitalCostsTab", "TechnicalEquipmentTab", "SoftwareTab"]
