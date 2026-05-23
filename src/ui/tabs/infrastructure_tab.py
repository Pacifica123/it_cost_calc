"""Legacy IT-infrastructure sandbox tab.

The tab is intentionally kept as an auxiliary free-form input area.  Structured
technical/software costs live in the ТО/ПО/OPEX tabs and are processed by the
application services.  Rows created here are namespaced as legacy sandbox data
so they do not accidentally become a second strict analytics model.
"""

from __future__ import annotations

import logging
import re
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from shared.constants import LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX
from shared.validation import parse_float, parse_int, require_text
from ui.tabs.base_scrollable_tab import BaseScrollableTab

logger = logging.getLogger(__name__)


class ITInfrastructureTab(BaseScrollableTab):
    """Free-form legacy tab for draft infrastructure articles.

    The tab remains available for quick notes and experimental custom articles,
    but it no longer presents itself as the main source of CAPEX/OPEX analytics.
    New rows are stored under ``legacy_infrastructure:*`` keys and marked with
    ``_legacy_sandbox`` metadata.  Strict calculations should use ТО, ПО, OPEX,
    energy, GA/AHP and DecisionReport flows instead.
    """

    COLUMN_NAMES = {
        "name": "Наименование",
        "quantity": "Количество",
        "monthly_cost": "Ежемесячная стоимость",
        "one_time_cost": "Разовая стоимость",
    }
    LEGACY_META_FIELDS = {
        "_legacy_sandbox",
        "_legacy_article_title",
        "_legacy_expense_type",
        "_legacy_note",
    }

    def __init__(self, parent, crud):
        super().__init__(parent)
        self.crud = crud
        self.tables: dict[str, tuple[tk.Frame, ttk.Treeview, tk.Frame]] = {}

        self.inner_frame.columnconfigure(0, weight=1)
        self._build_intro()
        self.create_article_button = tk.Button(
            self.inner_frame,
            text="Создать вспомогательную статью",
            command=self.create_article,
        )
        self.create_article_button.pack(fill="x", padx=10, pady=(0, 10))

        self._restore_existing_sandbox_tables()
        self.update_scrollregion()

    def _build_intro(self) -> None:
        intro = tk.Label(
            self.inner_frame,
            text=(
                "Старая вкладка ИТ-инфраструктуры оставлена как вспомогательная песочница. "
                "Она подходит для черновых пользовательских статей и заметок, но не является "
                "основным источником строгой аналитики. Для расчётов используйте вкладки ТО, ПО, "
                "Операционные затраты, Электроэнергия, GA/AHP, NPV и Экспорт."
            ),
            anchor="w",
            justify="left",
            wraplength=1080,
        )
        intro.pack(fill="x", padx=10, pady=(10, 6))

        hint = tk.Label(
            self.inner_frame,
            text=(
                "Новые записи этой вкладки сохраняются в runtime-хранилище с префиксом "
                f"{LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX} и не попадают в общий пул "
                "CandidateConfiguration без отдельной нормализации."
            ),
            anchor="w",
            justify="left",
            fg="#555555",
            wraplength=1080,
        )
        hint.pack(fill="x", padx=10, pady=(0, 10))

    def _restore_existing_sandbox_tables(self) -> None:
        for entity_key, rows in sorted(self.crud.entities.items()):
            if not self._is_sandbox_key(entity_key):
                continue
            title = self._title_from_rows(entity_key, rows)
            columns = self._columns_from_rows(rows)
            self._create_sandbox_table(entity_key, title, columns)
            self.crud.sync_table(entity_key, self.tables[entity_key][1])

    def create_article(self) -> None:
        def save_article() -> None:
            try:
                article_title = require_text(name_entry.get(), "Наименование статьи")
            except ValueError as error:
                messagebox.showerror("Ошибка ввода", str(error), parent=popup)
                return

            columns = ["name"]
            if quantity_var.get():
                columns.append("quantity")

            expense_type = expense_type_var.get()
            if expense_type == "periodic":
                columns.append("monthly_cost")
            elif expense_type == "one_time":
                columns.append("one_time_cost")

            entity_key = self._unique_sandbox_key(article_title)
            self._create_sandbox_table(entity_key, article_title, columns)
            self.update_scrollregion()
            popup.destroy()

        popup = tk.Toplevel(self)
        popup.title("Новая вспомогательная статья")
        popup.transient(self)
        popup.grab_set()

        container = tk.Frame(popup, padx=12, pady=12)
        container.pack(fill="both", expand=True)

        tk.Label(container, text="Наименование статьи").grid(row=0, column=0, sticky="w", pady=4)
        name_entry = tk.Entry(container, width=36)
        name_entry.grid(row=0, column=1, sticky="ew", pady=4)

        quantity_var = tk.BooleanVar()
        tk.Checkbutton(
            container,
            text="Учитывать количество",
            variable=quantity_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=4)

        tk.Label(container, text="Тип расходов").grid(row=2, column=0, sticky="w", pady=4)
        expense_type_var = tk.StringVar(value="periodic")
        ttk.Combobox(
            container,
            textvariable=expense_type_var,
            values=("periodic", "one_time", "none"),
            state="readonly",
            width=33,
        ).grid(row=2, column=1, sticky="ew", pady=4)

        buttons = tk.Frame(container)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(10, 0))
        tk.Button(buttons, text="Сохранить", command=save_article).pack(side="left", padx=(0, 8))
        tk.Button(buttons, text="Отмена", command=popup.destroy).pack(side="left")

        container.columnconfigure(1, weight=1)
        name_entry.focus_set()
        popup.bind("<Return>", lambda _event: save_article())
        popup.bind("<Escape>", lambda _event: popup.destroy())

    def _create_sandbox_table(
        self,
        entity_key: str,
        title: str,
        columns: list[str] | tuple[str, ...],
    ) -> None:
        if entity_key in self.tables:
            return

        frame = tk.Frame(self.inner_frame, padx=10, pady=8, relief="groove", borderwidth=1)
        frame.pack(fill="x", padx=10, pady=6)

        label = tk.Label(
            frame,
            text=f"{title} (вспомогательная статья)",
            anchor="w",
            justify="left",
            font=("TkDefaultFont", 10, "bold"),
        )
        label.pack(fill="x")

        table = ttk.Treeview(frame, columns=tuple(columns), show="headings", height=5)
        for column in columns:
            table.column(column, stretch=tk.YES)
            table.heading(column, text=self.COLUMN_NAMES.get(column, column))
        table.pack(fill="x", pady=(6, 4))

        buttons_frame = tk.Frame(frame)
        buttons_frame.pack(fill="x")
        tk.Button(
            buttons_frame,
            text="Добавить",
            command=lambda key=entity_key: self.add_row(key),
        ).pack(side="left")
        tk.Button(
            buttons_frame,
            text="Редактировать",
            command=lambda key=entity_key: self.edit_row(key),
        ).pack(side="left", padx=(6, 0))
        tk.Button(
            buttons_frame,
            text="Удалить",
            command=lambda key=entity_key: self.delete_row(key),
        ).pack(side="left", padx=(6, 0))

        self.tables[entity_key] = (frame, table, buttons_frame)

    def add_row(self, entity_key: str) -> None:
        table = self.tables[entity_key][1]
        self._show_row_dialog(entity_key, table, title="Добавить запись")

    def edit_row(self, entity_key: str) -> None:
        table = self.tables[entity_key][1]
        selected_index = self._selected_index(table)
        if selected_index is None:
            messagebox.showwarning("Нет выбора", "Выберите строку для редактирования", parent=self)
            return
        current_row = self.crud.entities[entity_key][selected_index]
        self._show_row_dialog(
            entity_key,
            table,
            title="Редактировать запись",
            initial_values=current_row,
            selected_index=selected_index,
        )

    def delete_row(self, entity_key: str) -> None:
        table = self.tables[entity_key][1]
        selected_index = self._selected_index(table)
        if selected_index is None:
            messagebox.showwarning("Нет выбора", "Выберите строку для удаления", parent=self)
            return
        self.crud.delete(entity_key, selected_index, table)

    def _show_row_dialog(
        self,
        entity_key: str,
        table: ttk.Treeview,
        *,
        title: str,
        initial_values: dict[str, Any] | None = None,
        selected_index: int | None = None,
    ) -> None:
        initial_values = initial_values or {}
        popup = tk.Toplevel(self)
        popup.title(title)
        popup.transient(self)
        popup.grab_set()

        container = tk.Frame(popup, padx=12, pady=12)
        container.pack(fill="both", expand=True)

        entries: dict[str, tk.Entry] = {}
        columns = tuple(table["columns"])
        for row_index, column in enumerate(columns):
            tk.Label(container, text=self.COLUMN_NAMES.get(column, column)).grid(
                row=row_index,
                column=0,
                sticky="w",
                pady=4,
            )
            entry = tk.Entry(container, width=36)
            entry.grid(row=row_index, column=1, sticky="ew", pady=4)
            entry.insert(0, str(initial_values.get(column, "")))
            entries[column] = entry

        def save_data() -> None:
            try:
                data = self._payload_from_entries(entity_key, entries)
            except ValueError as error:
                messagebox.showerror("Ошибка ввода", str(error), parent=popup)
                return

            if selected_index is None:
                self.crud.add(entity_key, data, table)
            else:
                self.crud.update(entity_key, selected_index, data, table)
            popup.destroy()

        buttons = tk.Frame(container)
        buttons.grid(row=len(columns), column=0, columnspan=2, sticky="e", pady=(10, 0))
        tk.Button(buttons, text="Сохранить", command=save_data).pack(side="left", padx=(0, 8))
        tk.Button(buttons, text="Отмена", command=popup.destroy).pack(side="left")

        container.columnconfigure(1, weight=1)
        if entries:
            next(iter(entries.values())).focus_set()
        popup.bind("<Return>", lambda _event: save_data())
        popup.bind("<Escape>", lambda _event: popup.destroy())

    def _payload_from_entries(self, entity_key: str, entries: dict[str, tk.Entry]) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for column, entry in entries.items():
            raw_value = entry.get()
            if column == "name":
                data[column] = require_text(raw_value, self.COLUMN_NAMES[column])
            elif column == "quantity":
                data[column] = parse_int(raw_value, self.COLUMN_NAMES[column])
            elif column in {"monthly_cost", "one_time_cost"}:
                data[column] = parse_float(raw_value, self.COLUMN_NAMES[column])
            else:
                data[column] = raw_value

        data.update(
            {
                "_legacy_sandbox": True,
                "_legacy_article_title": self._title_from_key(entity_key),
                "_legacy_expense_type": self._expense_type_from_columns(entries.keys()),
                "_legacy_note": (
                    "Вспомогательная запись старой вкладки ИТ-инфраструктуры; "
                    "не участвует в строгой аналитике без отдельной нормализации."
                ),
            }
        )
        return data

    def _selected_index(self, table: ttk.Treeview) -> int | None:
        selected_item = table.focus()
        if not selected_item:
            return None
        children = list(table.get_children())
        if selected_item not in children:
            return None
        return children.index(selected_item)

    def _columns_from_rows(self, rows: list[dict[str, Any]]) -> list[str]:
        columns = ["name"]
        visible_keys = {
            key
            for row in rows
            for key in dict(row).keys()
            if key not in self.LEGACY_META_FIELDS
        }
        if "quantity" in visible_keys:
            columns.append("quantity")
        if "monthly_cost" in visible_keys:
            columns.append("monthly_cost")
        elif "one_time_cost" in visible_keys:
            columns.append("one_time_cost")
        return columns

    def _expense_type_from_columns(self, columns) -> str:
        column_set = set(columns)
        if "monthly_cost" in column_set:
            return "periodic"
        if "one_time_cost" in column_set:
            return "one_time"
        return "none"

    def _title_from_rows(self, entity_key: str, rows: list[dict[str, Any]]) -> str:
        for row in rows:
            title = row.get("_legacy_article_title")
            if title:
                return str(title)
        return self._title_from_key(entity_key)

    def _title_from_key(self, entity_key: str) -> str:
        if entity_key.startswith(LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX):
            raw = entity_key.removeprefix(LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX)
        else:
            raw = entity_key
        return raw.replace("_", " ").strip() or "Вспомогательная статья"

    def _unique_sandbox_key(self, title: str) -> str:
        base = self._sandbox_key(title)
        candidate = base
        counter = 2
        while candidate in self.tables or candidate in self.crud.entities:
            candidate = f"{base}_{counter}"
            counter += 1
        return candidate

    def _sandbox_key(self, title: str) -> str:
        slug = re.sub(r"[^\w]+", "_", title.strip().lower(), flags=re.UNICODE).strip("_")
        return f"{LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX}{slug or 'article'}"

    def _is_sandbox_key(self, entity_key: str) -> bool:
        return str(entity_key).startswith(LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX)


__all__ = ["ITInfrastructureTab"]
