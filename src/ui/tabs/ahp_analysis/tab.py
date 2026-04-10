"""Refactored AHP analysis tab assembled from smaller mixins."""

import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from .config_mixin import AHPConfigurationMixin
from .io_mixin import AHPIOMixin
from .presenter_mixin import AHPPresenterMixin
from .constants import (
    DEFAULT_CONSTRAINTS,
    DEFAULT_SOFT_CRITERIA,
    SOFT_CRITERIA_LABELS,
    SOFT_CRITERIA_SHORT_LABELS,
)
from .matrix_utils import (
    ALLOWED_PAIRWISE_HINTS,
    build_pairwise_matrix,
    default_pairwise_value,
    extract_default_expert_matrices,
    format_pairwise_value,
    parse_pairwise_value,
)

try:
    from domain.decision.ahp.configuration_selector import run_ahp_pipeline
except Exception:
    run_ahp_pipeline = None


class ConfigurationSelectionTab(AHPConfigurationMixin, AHPIOMixin, AHPPresenterMixin, tk.Frame):
    def __init__(self, parent, crud):
        super().__init__(parent)
        self.crud = crud
        self.configurations: dict[str, dict] = {}
        self.soft_criteria_vars = {
            criterion_id: tk.BooleanVar(value=True) for criterion_id in DEFAULT_SOFT_CRITERIA
        }
        self.trust_var = tk.IntVar(value=1)
        self.expert_mode_hint_var = tk.StringVar()
        self.custom_pairwise_vars: dict[tuple[str, str], tk.StringVar] = {}
        self.reciprocal_pairwise_vars: dict[tuple[str, str], tk.StringVar] = {}
        self.matrix_entries: list[tk.Entry] = []

        self._build_ui()
        self._reset_custom_matrix_to_defaults()
        self._rebuild_matrix_editor()
        self._sync_trust_mode_state()
        self._set_empty_state_message()

        repo_root = Path(__file__).resolve().parents[4]
        self.reports_dir = repo_root / "data" / "generated" / "ahp"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _build_ui(self):
        ctrl = tk.Frame(self)
        ctrl.pack(fill="x", padx=6, pady=6)

        tk.Button(ctrl, text="Добавить конфигурацию", command=self._add_config_popup).pack(
            side="left"
        )
        tk.Button(ctrl, text="Удалить выбранную", command=self._delete_selected_config).pack(
            side="left"
        )
        tk.Button(ctrl, text="Загрузить из JSON...", command=self._load_from_json).pack(side="left")
        tk.Button(
            ctrl, text="Сохранить конфигурации в JSON...", command=self._save_configs_to_json
        ).pack(side="left")

        self.cfg_table = ttk.Treeview(
            self,
            columns=("id", "name", "people", "devices_count"),
            show="headings",
            height=6,
        )
        self.cfg_table.heading("id", text="ID")
        self.cfg_table.heading("name", text="Название")
        self.cfg_table.heading("people", text="People")
        self.cfg_table.heading("devices_count", text="Devices")
        self.cfg_table.column("id", width=120, anchor="center")
        self.cfg_table.column("name", width=250, anchor="w")
        self.cfg_table.column("people", width=80, anchor="center")
        self.cfg_table.column("devices_count", width=80, anchor="center")
        self.cfg_table.pack(fill="x", padx=6, pady=(0, 6))
        self.cfg_table.bind("<<TreeviewSelect>>", self._on_select_config)

        dev_frame = tk.Frame(self)
        dev_frame.pack(fill="x", padx=6, pady=6)

        tk.Label(dev_frame, text="Устройства выбранной конфигурации:").pack(anchor="w")
        self.dev_table = ttk.Treeview(
            dev_frame,
            columns=("role", "vendor", "cpu", "ram", "energy", "cost", "rel_low", "rel_high"),
            show="headings",
            height=6,
        )
        for col, name in [
            ("role", "Role"),
            ("vendor", "Vendor / Source"),
            ("cpu", "CPU"),
            ("ram", "RAM"),
            ("energy", "Energy"),
            ("cost", "Cost"),
            ("rel_low", "Rel low"),
            ("rel_high", "Rel high"),
        ]:
            self.dev_table.heading(col, text=name)
        self.dev_table.pack(fill="x")

        dev_btns = tk.Frame(self)
        dev_btns.pack(fill="x", padx=6)
        tk.Button(dev_btns, text="Добавить устройство", command=self._add_device_popup).pack(
            side="left"
        )
        tk.Button(dev_btns, text="Удалить устройство", command=self._delete_device).pack(
            side="left"
        )

        conf_frame = tk.LabelFrame(self, text="Параметры анализа")
        conf_frame.pack(fill="x", padx=6, pady=6)

        tk.Label(conf_frame, text="Max budget:").grid(row=0, column=0, sticky="w")
        self.e_budget = tk.Entry(conf_frame, width=14)
        self.e_budget.grid(row=0, column=1, sticky="w")
        self.e_budget.insert(0, str(DEFAULT_CONSTRAINTS["max_budget"]))
        tk.Label(conf_frame, text="Max energy:").grid(row=0, column=2, sticky="w")
        self.e_energy = tk.Entry(conf_frame, width=14)
        self.e_energy.grid(row=0, column=3, sticky="w")
        self.e_energy.insert(0, str(DEFAULT_CONSTRAINTS["max_energy"]))
        tk.Label(conf_frame, text="People tolerance (0..1):").grid(row=0, column=4, sticky="w")
        self.e_tol = tk.Entry(conf_frame, width=10)
        self.e_tol.grid(row=0, column=5, sticky="w")
        self.e_tol.insert(0, str(DEFAULT_CONSTRAINTS["people_match_tolerance"]))

        tk.Label(conf_frame, text="Критерии для расчета:").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        criteria_frame = tk.Frame(conf_frame)
        criteria_frame.grid(row=2, column=0, columnspan=6, sticky="w", pady=(4, 0))
        for idx, criterion_id in enumerate(DEFAULT_SOFT_CRITERIA):
            ttk.Checkbutton(
                criteria_frame,
                text=self._format_soft_criterion_label(criterion_id),
                variable=self.soft_criteria_vars[criterion_id],
                command=self._on_selected_criteria_changed,
            ).grid(row=idx // 2, column=idx % 2, sticky="w", padx=(0, 18), pady=2)

        self.chk_trust = tk.Checkbutton(
            conf_frame,
            text="Полностью довериться машине",
            variable=self.trust_var,
            command=self._on_trust_mode_changed,
        )
        self.chk_trust.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))

        tk.Button(
            conf_frame,
            text="Рассчитать оптимальную конфигурацию",
            command=self._run_analysis,
        ).grid(row=3, column=3, columnspan=2, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(
            conf_frame,
            textvariable=self.expert_mode_hint_var,
            wraplength=850,
            justify="left",
        ).grid(row=4, column=0, columnspan=6, sticky="w", pady=(6, 0))

        self.matrix_box = tk.LabelFrame(
            conf_frame,
            text="Экспертная матрица важности критериев",
        )
        self.matrix_box.grid(row=5, column=0, columnspan=6, sticky="ew", pady=(8, 0))
        self.matrix_box.grid_columnconfigure(0, weight=1)

        matrix_hint = tk.Label(
            self.matrix_box,
            text=(
                "В экспертном режиме задайте попарные сравнения для включенных критериев. "
                f"Можно вводить числа или дроби вида 1/3. Подсказка по шкале: {', '.join(ALLOWED_PAIRWISE_HINTS)}."
            ),
            justify="left",
            anchor="w",
        )
        matrix_hint.pack(fill="x", padx=6, pady=(6, 4))

        matrix_tools = tk.Frame(self.matrix_box)
        matrix_tools.pack(fill="x", padx=6, pady=(0, 4))
        tk.Button(
            matrix_tools,
            text="Сбросить к машинной подсказке",
            command=self._reset_custom_matrix_to_defaults,
        ).pack(side="left")

        self.matrix_grid = tk.Frame(self.matrix_box)
        self.matrix_grid.pack(fill="x", padx=6, pady=(0, 6))

        out_frame = tk.LabelFrame(self, text="Результат")
        out_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.summary_text = tk.Text(out_frame, height=10)
        self.summary_text.pack(fill="both", expand=False)

        self.details_shown = False
        self.details_btn = tk.Button(
            out_frame, text="Показать детали (CI/CR, матрицы, веса)", command=self._toggle_details
        )
        self.details_btn.pack(pady=(6, 0))
        self.details_panel = tk.Text(out_frame, height=12)

    def _set_empty_state_message(self):
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert(
            "1.0",
            "Сначала загрузите конфигурации вручную из JSON или общей кнопкой \"Загрузить демо-данные\".",
        )

    def _format_soft_criterion_label(self, criterion_id: str) -> str:
        return SOFT_CRITERIA_LABELS.get(criterion_id, criterion_id)

    def _format_soft_criterion_short_label(self, criterion_id: str) -> str:
        return SOFT_CRITERIA_SHORT_LABELS.get(criterion_id, self._format_soft_criterion_label(criterion_id))

    def _configuration_display_name(self, config_id: str) -> str:
        config = self.configurations.get(config_id, {})
        return config.get("name", config_id)

    def _selected_soft_criteria(self) -> list[str]:
        return [
            criterion_id
            for criterion_id in DEFAULT_SOFT_CRITERIA
            if self.soft_criteria_vars[criterion_id].get()
        ]

    def _pair_key(self, left_id: str, right_id: str) -> tuple[str, str]:
        left_idx = DEFAULT_SOFT_CRITERIA.index(left_id)
        right_idx = DEFAULT_SOFT_CRITERIA.index(right_id)
        if left_idx < right_idx:
            return left_id, right_id
        return right_id, left_id

    def _pairwise_var(self, left_id: str, right_id: str) -> tk.StringVar:
        key = self._pair_key(left_id, right_id)
        if key not in self.custom_pairwise_vars:
            self.custom_pairwise_vars[key] = tk.StringVar(
                value=format_pairwise_value(default_pairwise_value(*key))
            )
        return self.custom_pairwise_vars[key]

    def _reciprocal_var(self, left_id: str, right_id: str) -> tk.StringVar:
        key = self._pair_key(left_id, right_id)
        if key not in self.reciprocal_pairwise_vars:
            self.reciprocal_pairwise_vars[key] = tk.StringVar()
        self._update_reciprocal_pair(key)
        return self.reciprocal_pairwise_vars[key]

    def _update_reciprocal_pair(self, key: tuple[str, str]) -> None:
        reciprocal_var = self.reciprocal_pairwise_vars.setdefault(key, tk.StringVar())
        try:
            value = parse_pairwise_value(self.custom_pairwise_vars[key].get())
            reciprocal_var.set(format_pairwise_value(1.0 / value))
        except Exception:
            reciprocal_var.set("—")

    def _reset_custom_matrix_to_defaults(self) -> None:
        self.custom_pairwise_vars.clear()
        self.reciprocal_pairwise_vars.clear()
        selected = self._selected_soft_criteria() or list(DEFAULT_SOFT_CRITERIA)
        for row_index, left_id in enumerate(selected):
            for col_index in range(row_index + 1, len(selected)):
                right_id = selected[col_index]
                self.custom_pairwise_vars[(left_id, right_id)] = tk.StringVar(
                    value=format_pairwise_value(default_pairwise_value(left_id, right_id))
                )
        self._rebuild_matrix_editor()

    def _sanitize_pairwise_var(self, left_id: str, right_id: str) -> None:
        var = self._pairwise_var(left_id, right_id)
        try:
            value = parse_pairwise_value(var.get())
        except Exception as exc:
            messagebox.showerror("Ошибка ввода", str(exc), parent=self)
            value = default_pairwise_value(left_id, right_id)
        var.set(format_pairwise_value(value))
        self._update_reciprocal_pair((left_id, right_id))

    def _on_selected_criteria_changed(self) -> None:
        self._rebuild_matrix_editor()
        self._sync_trust_mode_state()

    def _on_trust_mode_changed(self) -> None:
        self._sync_trust_mode_state()

    def _sync_trust_mode_state(self) -> None:
        machine_mode = bool(self.trust_var.get())
        if machine_mode:
            self.expert_mode_hint_var.set(
                "Машинный режим включен: для выбранных критериев будут использованы встроенные экспертные матрицы текущей модели."
            )
        else:
            self.expert_mode_hint_var.set(
                "Экспертный режим включен: ниже можно вручную задать матрицу важности критериев."
            )
        state = "disabled" if machine_mode else "normal"
        for entry in self.matrix_entries:
            entry.configure(state=state)

    def _rebuild_matrix_editor(self) -> None:
        for widget in self.matrix_grid.winfo_children():
            widget.destroy()
        self.matrix_entries.clear()

        selected = self._selected_soft_criteria()
        if len(selected) < 2:
            ttk.Label(
                self.matrix_grid,
                text="Для построения матрицы выберите минимум два критерия.",
            ).grid(row=0, column=0, sticky="w")
            return

        ttk.Label(self.matrix_grid, text="").grid(row=0, column=0, padx=4, pady=2)
        for col_index, criterion_id in enumerate(selected, start=1):
            ttk.Label(
                self.matrix_grid,
                text=self._format_soft_criterion_short_label(criterion_id),
                anchor="center",
            ).grid(row=0, column=col_index, padx=4, pady=2)

        for row_index, row_id in enumerate(selected, start=1):
            ttk.Label(
                self.matrix_grid,
                text=self._format_soft_criterion_short_label(row_id),
                anchor="w",
            ).grid(row=row_index, column=0, padx=4, pady=2, sticky="w")
            for col_index, col_id in enumerate(selected, start=1):
                if row_index == col_index:
                    ttk.Label(self.matrix_grid, text="1", anchor="center", width=10).grid(
                        row=row_index, column=col_index, padx=2, pady=2
                    )
                elif row_index < col_index:
                    entry = ttk.Entry(
                        self.matrix_grid,
                        width=10,
                        textvariable=self._pairwise_var(row_id, col_id),
                    )
                    entry.grid(row=row_index, column=col_index, padx=2, pady=2)
                    entry.bind(
                        "<FocusOut>",
                        lambda _event, left=row_id, right=col_id: self._sanitize_pairwise_var(
                            left, right
                        ),
                    )
                    entry.bind(
                        "<Return>",
                        lambda _event, left=row_id, right=col_id: self._sanitize_pairwise_var(
                            left, right
                        ),
                    )
                    self.matrix_entries.append(entry)
                else:
                    ttk.Label(
                        self.matrix_grid,
                        textvariable=self._reciprocal_var(col_id, row_id),
                        anchor="center",
                        width=10,
                    ).grid(row=row_index, column=col_index, padx=2, pady=2)

    def _replace_configurations(self, cfgs) -> None:
        self.configurations.clear()
        for item in self.cfg_table.get_children():
            self.cfg_table.delete(item)
        self.dev_table.delete(*self.dev_table.get_children())

        for config in cfgs:
            cid = config.get("id") or config.get("name") or f"cfg_{len(self.configurations) + 1}"
            name = config.get("name") or cid
            devices = config.get("devices", [])
            people = int(config.get("meta", {}).get("people", config.get("people", 0)))
            self.configurations[cid] = {
                "name": name,
                "devices": devices,
                "meta": {"people": people},
            }
            self.cfg_table.insert("", "end", iid=cid, values=(cid, name, people, len(devices)))

        children = self.cfg_table.get_children()
        if children:
            self.cfg_table.selection_set(children[0])
            self._on_select_config(None)

    def load_demo_configurations(self, configurations, *, constraints=None, message=None) -> None:
        self._replace_configurations(configurations)
        if constraints:
            self._replace_entry_value(self.e_budget, constraints.get("max_budget"))
            self._replace_entry_value(self.e_energy, constraints.get("max_energy"))
            self._replace_entry_value(
                self.e_tol, constraints.get("people_match_tolerance", DEFAULT_CONSTRAINTS["people_match_tolerance"])
            )

        for criterion_id in DEFAULT_SOFT_CRITERIA:
            self.soft_criteria_vars[criterion_id].set(True)
        self.trust_var.set(1)
        self._reset_custom_matrix_to_defaults()
        self._sync_trust_mode_state()

        self.summary_text.delete("1.0", "end")
        self.summary_text.insert(
            "1.0",
            message
            or "Демонстрационные конфигурации загружены из общего набора данных. При необходимости измените ограничения и запустите расчет.",
        )
        if self.details_shown:
            self.details_panel.delete("1.0", "end")

    def _replace_entry_value(self, entry: tk.Entry, value) -> None:
        entry.delete(0, "end")
        entry.insert(0, "" if value is None else str(value))

    def _collect_expert_matrices(self, soft_criteria: list[str]):
        if len(soft_criteria) < 2:
            raise ValueError("Для AHP-анализа нужно выбрать минимум два критерия")

        if self.trust_var.get():
            return extract_default_expert_matrices(soft_criteria)

        matrix = build_pairwise_matrix(
            soft_criteria,
            lambda left_id, right_id: self._pairwise_var(left_id, right_id).get(),
        )
        return [matrix, matrix.copy()]

    def _run_analysis(self):
        if run_ahp_pipeline is None:
            messagebox.showerror(
                "Ошибка",
                "AHP engine не доступен. Проверьте модуль domain.decision.ahp.configuration_selector",
                parent=self,
            )
            return

        configs = []
        for cid, value in self.configurations.items():
            configs.append(
                {
                    "id": cid,
                    "name": value.get("name", cid),
                    "devices": value["devices"],
                    "meta": value.get("meta", {}),
                }
            )

        if not configs:
            messagebox.showerror("Ошибка", "Нет конфигураций для анализа", parent=self)
            return

        try:
            constraints = {
                "max_budget": float(self.e_budget.get()),
                "max_energy": float(self.e_energy.get()),
                "people_match_tolerance": float(self.e_tol.get()),
            }
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Параметры constraints неверны: {exc}", parent=self)
            return

        soft_criteria = self._selected_soft_criteria()
        try:
            experts = self._collect_expert_matrices(soft_criteria)
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc), parent=self)
            return

        try:
            report = run_ahp_pipeline(
                configurations=configs,
                soft_criteria=soft_criteria,
                experts_criteria_matrices=experts,
                constraints=constraints,
                saaty_cap=True,
                top_pct=0.10,
            )
        except Exception as exc:
            messagebox.showerror("Ошибка при выполнении AHP", str(exc), parent=self)
            return

        out_path = self.reports_dir / "ahp_report_sample.json"
        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            messagebox.showwarning("Warning", f"Не удалось сохранить отчёт: {exc}", parent=self)

        self._display_summary(report)
        self._fill_details(report)


AHPAnalysisTab = ConfigurationSelectionTab

__all__ = ["ConfigurationSelectionTab", "AHPAnalysisTab"]
