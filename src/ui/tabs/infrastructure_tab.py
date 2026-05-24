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
from typing import Any, Callable

from application.services.solution_component_sandbox_conversion_service import (
    SolutionComponentSandboxConversionService,
)
from application.services.solution_component_runtime_service import SolutionComponentRuntimeService
from domain import ComponentType
from shared.constants import LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX
from shared.validation import parse_float, parse_int, require_text
from ui.tabs.base_scrollable_tab import BaseScrollableTab
from ui.tabs.solution_component_editor_tab import format_normalization_preview

logger = logging.getLogger(__name__)

_CONVERSION_SCOPE_COMPONENT_TYPES = {
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

_CONVERSION_FINANCIAL_FIELDS = (
    ("purchase_cost", "Покупка / CAPEX"),
    ("implementation_cost", "Внедрение"),
    ("migration_cost", "Миграция"),
    ("testing_cost", "Тестирование"),
    ("monthly_cost", "Ежемесячно"),
    ("annual_cost", "Ежегодно"),
    ("energy_cost", "Энергия / мес."),
    ("quantity", "Количество"),
)

_CONVERSION_METRIC_FIELDS = (
    ("max_power", "Мощность, Вт"),
    ("client_seats", "Рабочие места"),
    ("license_units", "Лицензии / пользователи"),
    ("functionality_score", "Функциональность"),
    ("performance_score", "Производительность"),
    ("reliability_score", "Надёжность"),
    ("support_score", "Поддержка"),
    ("hours_per_day", "Часы работы в день"),
    ("working_days", "Рабочие дни в месяце"),
)


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

    def __init__(
        self,
        parent,
        crud,
        *,
        solution_component_runtime_service: SolutionComponentRuntimeService | None = None,
        conversion_service: SolutionComponentSandboxConversionService | None = None,
        on_component_converted: Callable[[], None] | None = None,
    ):
        super().__init__(parent)
        self.crud = crud
        self.solution_component_runtime_service = solution_component_runtime_service
        self.conversion_service = conversion_service or SolutionComponentSandboxConversionService()
        self.on_component_converted = on_component_converted
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
        if self.solution_component_runtime_service is not None:
            tk.Button(
                buttons_frame,
                text="Преобразовать в компонент",
                command=lambda key=entity_key: self.convert_selected_row(key),
            ).pack(side="left", padx=(12, 0))

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


    def convert_selected_row(self, entity_key: str) -> None:
        """Open explicit conversion dialog for one selected legacy row."""

        if self.solution_component_runtime_service is None:
            messagebox.showwarning(
                "Конвертация недоступна",
                "Редактор компонентов не подключён к этой вкладке.",
                parent=self,
            )
            return
        table = self.tables[entity_key][1]
        selected_index = self._selected_index(table)
        if selected_index is None:
            messagebox.showwarning("Нет выбора", "Выберите sandbox-запись для конвертации", parent=self)
            return
        rows = self.crud.entities.get(entity_key, [])
        if selected_index >= len(rows):
            messagebox.showwarning("Нет строки", "Выбранная sandbox-запись уже недоступна.", parent=self)
            return
        self._show_conversion_dialog(entity_key, selected_index, rows[selected_index])

    def _show_conversion_dialog(
        self,
        entity_key: str,
        selected_index: int,
        row: dict[str, Any],
    ) -> None:
        draft = self.conversion_service.build_conversion_draft(
            row,
            sandbox_entity=entity_key,
            row_index=selected_index,
        )
        popup = tk.Toplevel(self)
        popup.title("Преобразовать sandbox-запись в компонент")
        popup.transient(self)
        popup.grab_set()

        container = tk.Frame(popup, padx=12, pady=12)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)
        container.columnconfigure(3, weight=1)

        intro = tk.Label(
            container,
            text=(
                "Конвертация выполняется только вручную. Исходная sandbox-запись останется на месте; "
                "новая строка попадёт в редактор компонентов как черновик, пока вы явно не включите "
                "строгую аналитику и не заполните обязательные поля."
            ),
            justify="left",
            anchor="w",
            wraplength=820,
            fg="#555555",
        )
        intro.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 8))

        id_var = tk.StringVar(value=str(draft.get("id", "")))
        name_var = tk.StringVar(value=str(draft.get("name", "")))
        scope_var = tk.StringVar(value=str(draft.get("scope") or "mixed"))
        type_var = tk.StringVar(value=str(draft.get("component_type") or ComponentType.BUNDLE.value))
        strict_var = tk.BooleanVar(value=bool(draft.get("strict_analysis_participation", False)))

        tk.Label(container, text="ID компонента").grid(row=1, column=0, sticky="w", pady=3)
        tk.Entry(container, textvariable=id_var, width=36).grid(row=1, column=1, sticky="ew", pady=3)
        tk.Label(container, text="Название").grid(row=2, column=0, sticky="w", pady=3)
        tk.Entry(container, textvariable=name_var, width=36).grid(row=2, column=1, sticky="ew", pady=3)

        tk.Label(container, text="Профиль").grid(row=1, column=2, sticky="w", padx=(12, 4), pady=3)
        scope_combo = ttk.Combobox(
            container,
            textvariable=scope_var,
            values=tuple(_CONVERSION_SCOPE_COMPONENT_TYPES),
            state="readonly",
            width=26,
        )
        scope_combo.grid(row=1, column=3, sticky="ew", pady=3)
        tk.Label(container, text="Тип компонента").grid(row=2, column=2, sticky="w", padx=(12, 4), pady=3)
        type_combo = ttk.Combobox(
            container,
            textvariable=type_var,
            values=_CONVERSION_SCOPE_COMPONENT_TYPES.get(scope_var.get(), _CONVERSION_SCOPE_COMPONENT_TYPES["mixed"]),
            state="readonly",
            width=26,
        )
        type_combo.grid(row=2, column=3, sticky="ew", pady=3)

        def on_scope_change(_event=None) -> None:
            allowed = _CONVERSION_SCOPE_COMPONENT_TYPES.get(scope_var.get(), _CONVERSION_SCOPE_COMPONENT_TYPES["mixed"])
            type_combo.configure(values=allowed)
            if type_var.get() not in allowed:
                type_var.set(allowed[0])

        scope_combo.bind("<<ComboboxSelected>>", on_scope_change)

        finance_frame = tk.LabelFrame(container, text="Разнести сумму по финансовым полям", padx=8, pady=6)
        finance_frame.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        finance_frame.columnconfigure(1, weight=1)
        finance_frame.columnconfigure(3, weight=1)
        financial_vars: dict[str, tk.StringVar] = {}
        for idx, (key, label) in enumerate(_CONVERSION_FINANCIAL_FIELDS):
            row_no = idx // 2
            col_no = 0 if idx % 2 == 0 else 2
            var = tk.StringVar(value=_format_conversion_number(draft.get(key, "")))
            financial_vars[key] = var
            tk.Label(finance_frame, text=label).grid(row=row_no, column=col_no, sticky="w", padx=(0, 4), pady=3)
            tk.Entry(finance_frame, textvariable=var, width=24).grid(row=row_no, column=col_no + 1, sticky="ew", pady=3)

        metrics_frame = tk.LabelFrame(container, text="Метрики профиля (можно оставить пустыми)", padx=8, pady=6)
        metrics_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        metrics_frame.columnconfigure(1, weight=1)
        metrics_frame.columnconfigure(3, weight=1)
        metric_vars: dict[str, tk.StringVar] = {}
        for idx, (key, label) in enumerate(_CONVERSION_METRIC_FIELDS):
            row_no = idx // 2
            col_no = 0 if idx % 2 == 0 else 2
            value = dict(draft.get("metrics") or {}).get(key, "")
            var = tk.StringVar(value=_format_conversion_number(value))
            metric_vars[key] = var
            tk.Label(metrics_frame, text=label).grid(row=row_no, column=col_no, sticky="w", padx=(0, 4), pady=3)
            tk.Entry(metrics_frame, textvariable=var, width=24).grid(row=row_no, column=col_no + 1, sticky="ew", pady=3)

        assumptions_frame = tk.LabelFrame(container, text="Комментарий / допущения", padx=8, pady=6)
        assumptions_frame.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        assumptions_frame.columnconfigure(0, weight=1)
        assumptions_text = tk.Text(assumptions_frame, height=4, wrap="word")
        assumptions_text.grid(row=0, column=0, sticky="ew")
        assumptions_text.insert("1.0", "\n".join(draft.get("cost_assumptions") or []))

        tk.Checkbutton(
            container,
            text="Сразу включить в строгую аналитику после нормализации",
            variable=strict_var,
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(6, 2))

        preview_var = tk.StringVar(value="Нажмите «Предпросмотр», чтобы увидеть warnings до сохранения.")
        preview_label = tk.Label(container, textvariable=preview_var, justify="left", anchor="w", wraplength=820)
        preview_label.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(8, 4))

        def collect_overrides() -> dict[str, Any]:
            metrics = {
                key: _conversion_number_or_text(var.get())
                for key, var in metric_vars.items()
                if str(var.get()).strip() != ""
            }
            if str(financial_vars["quantity"].get()).strip():
                metrics.setdefault("quantity", _conversion_number_or_text(financial_vars["quantity"].get()))
            return {
                "id": id_var.get().strip(),
                "name": name_var.get().strip(),
                "scope": scope_var.get(),
                "component_type": type_var.get(),
                "strict_analysis_participation": strict_var.get(),
                **{key: var.get() for key, var in financial_vars.items()},
                "metrics": metrics,
                "cost_assumptions": assumptions_text.get("1.0", "end").splitlines(),
            }

        def preview() -> None:
            try:
                normalized = self.conversion_service.preview_conversion(
                    row,
                    sandbox_entity=entity_key,
                    row_index=selected_index,
                    overrides=collect_overrides(),
                )
            except ValueError as error:
                messagebox.showerror("Ошибка ввода", str(error), parent=popup)
                return
            preview_var.set(format_normalization_preview(normalized))

        def save() -> None:
            try:
                payload = self.conversion_service.build_conversion_draft(
                    row,
                    sandbox_entity=entity_key,
                    row_index=selected_index,
                    overrides=collect_overrides(),
                )
                normalized = self.solution_component_runtime_service.add_component(payload)
            except ValueError as error:
                messagebox.showerror("Ошибка ввода", str(error), parent=popup)
                return
            preview_var.set(format_normalization_preview(normalized))
            logger.info(
                "Sandbox-запись вручную преобразована в SolutionComponent: entity=%s index=%s component=%s",
                entity_key,
                selected_index,
                normalized.id,
            )
            if self.on_component_converted is not None:
                self.on_component_converted()
            messagebox.showinfo(
                "Компонент создан",
                "Новая строка добавлена в редактор компонентов. Исходная sandbox-запись не изменена.",
                parent=popup,
            )
            popup.destroy()

        buttons = tk.Frame(container)
        buttons.grid(row=8, column=0, columnspan=4, sticky="e", pady=(10, 0))
        tk.Button(buttons, text="Предпросмотр", command=preview).pack(side="left", padx=(0, 8))
        tk.Button(buttons, text="Создать компонент", command=save).pack(side="left", padx=(0, 8))
        tk.Button(buttons, text="Отмена", command=popup.destroy).pack(side="left")

        popup.bind("<Escape>", lambda _event: popup.destroy())
        on_scope_change()
        preview()

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


def _format_conversion_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value or "")
    if number.is_integer():
        return str(int(number))
    return str(number)


def _conversion_number_or_text(value: Any) -> Any:
    if value is None:
        return ""
    raw = str(value).strip()
    if raw == "":
        return ""
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return raw


__all__ = ["ITInfrastructureTab"]
