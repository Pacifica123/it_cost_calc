"""Refactored AHP analysis tab assembled from smaller mixins."""

import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from .config_mixin import AHPConfigurationMixin
from .io_mixin import AHPIOMixin
from .presenter_mixin import AHPPresenterMixin
from .constants import DEFAULT_SOFT_CRITERIA, M1, M2

try:
    from it_cost_calc.domain.decision.ahp.configuration_selector import (
        run_ahp_pipeline,
        aggregate_configuration,
    )
except Exception:
    run_ahp_pipeline = None
    aggregate_configuration = None


class ConfigurationSelectionTab(AHPConfigurationMixin, AHPIOMixin, AHPPresenterMixin, tk.Frame):
    def __init__(self, parent, crud):
        super().__init__(parent)
        self.crud = crud

        # Внутренние данные: словарь конфигураций id -> {'devices': [...], 'meta': {...}}
        self.configurations = {}

        # Верх: панель управления конфигурациями
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

        # Таблица конфигураций
        self.cfg_table = ttk.Treeview(
            self, columns=("id", "people", "devices_count"), show="headings", height=6
        )
        self.cfg_table.heading("id", text="ID")
        self.cfg_table.heading("people", text="People")
        self.cfg_table.heading("devices_count", text="Devices")
        self.cfg_table.pack(fill="x", padx=6, pady=(0, 6))
        self.cfg_table.bind("<<TreeviewSelect>>", self._on_select_config)

        # Секция — список устройств выбранной конфигурации
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
            ("vendor", "Vendor"),
            ("cpu", "CPU"),
            ("ram", "RAM"),
            ("energy", "Energy"),
            ("cost", "Cost"),
            ("rel_low", "Rel low"),
            ("rel_high", "Rel high"),
        ]:
            self.dev_table.heading(col, text=name)
        self.dev_table.pack(fill="x")
        # Кнопки для устройств
        dev_btns = tk.Frame(self)
        dev_btns.pack(fill="x", padx=6)
        tk.Button(dev_btns, text="Добавить устройство", command=self._add_device_popup).pack(
            side="left"
        )
        tk.Button(dev_btns, text="Удалить устройство", command=self._delete_device).pack(
            side="left"
        )

        # Constraints и запуск
        conf_frame = tk.LabelFrame(self, text="Параметры анализа")
        conf_frame.pack(fill="x", padx=6, pady=6)

        tk.Label(conf_frame, text="Max budget:").grid(row=0, column=0, sticky="w")
        self.e_budget = tk.Entry(conf_frame, width=12)
        self.e_budget.grid(row=0, column=1, sticky="w")
        self.e_budget.insert(0, "6000")
        tk.Label(conf_frame, text="Max energy:").grid(row=0, column=2, sticky="w")
        self.e_energy = tk.Entry(conf_frame, width=12)
        self.e_energy.grid(row=0, column=3, sticky="w")
        self.e_energy.insert(0, "700")
        tk.Label(conf_frame, text="People tolerance (0..1):").grid(row=0, column=4, sticky="w")
        self.e_tol = tk.Entry(conf_frame, width=8)
        self.e_tol.grid(row=0, column=5, sticky="w")
        self.e_tol.insert(0, "0.25")

        # Soft criteria editable field (comma-separated) — по умолчанию DEFAULT_SOFT_CRITERIA
        tk.Label(conf_frame, text="Soft criteria (comma-separated):").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        self.e_soft = tk.Entry(conf_frame, width=60)
        self.e_soft.grid(row=1, column=2, columnspan=4, sticky="w", pady=(6, 0))
        self.e_soft.insert(0, ",".join(DEFAULT_SOFT_CRITERIA))

        # Галка доверия (всегда включена и disabled)
        self.trust_var = tk.IntVar(value=1)
        self.chk_trust = tk.Checkbutton(
            conf_frame,
            text="Полностью довериться машине (по-умолчанию, недоступно)",
            variable=self.trust_var,
        )
        self.chk_trust.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.chk_trust.configure(state="disabled")

        # Run button
        run_btn = tk.Button(
            conf_frame, text="Рассчитать оптимальную конфигурацию", command=self._run_analysis
        )
        run_btn.grid(row=2, column=3, columnspan=2, sticky="w", padx=(6, 0), pady=(6, 0))

        # Output area: краткий и скрываемый подробный
        out_frame = tk.LabelFrame(self, text="Результат")
        out_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.summary_text = tk.Text(out_frame, height=10)
        self.summary_text.pack(fill="both", expand=False)

        # Toggle for details
        self.details_shown = False
        self.details_btn = tk.Button(
            out_frame, text="Показать детали (CI/CR, матрицы, веса)", command=self._toggle_details
        )
        self.details_btn.pack(pady=(6, 0))
        self.details_panel = tk.Text(out_frame, height=12)
        # по умолчанию скрыт

        # generated reports live under the repository data directory
        repo_root = Path(__file__).resolve().parents[5]
        self.reports_dir = repo_root / "data" / "generated" / "ahp"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _run_analysis(self):
        if run_ahp_pipeline is None:
            messagebox.showerror(
                "Ошибка",
                "AHP engine не доступен. Проверьте модуль domain.decision.ahp.configuration_selector",
            )
            return

        # Build configs list in expected format
        configs = []
        for cid, v in self.configurations.items():
            configs.append({"id": cid, "devices": v["devices"], "meta": v.get("meta", {})})

        # constraints from UI
        try:
            constraints = {
                "max_budget": float(self.e_budget.get()),
                "max_energy": float(self.e_energy.get()),
                "people_match_tolerance": float(self.e_tol.get()),
            }
        except Exception as e:
            messagebox.showerror("Ошибка", f"Параметры constraints неверны: {e}")
            return

        soft_criteria = [s.strip() for s in self.e_soft.get().split(",") if s.strip()]
        if len(soft_criteria) == 0:
            soft_criteria = DEFAULT_SOFT_CRITERIA

        # Use embedded expert matrices (M1,M2)
        experts = [M1, M2]

        # call pipeline
        try:
            report = run_ahp_pipeline(
                configurations=configs,
                soft_criteria=soft_criteria,
                experts_criteria_matrices=experts,
                constraints={
                    "max_budget": constraints["max_budget"],
                    "max_energy": constraints["max_energy"],
                    "people_match_tolerance": constraints["people_match_tolerance"],
                },
                saaty_cap=True,
                top_pct=0.10,
            )
        except Exception as e:
            messagebox.showerror("Ошибка при выполнении AHP", str(e))
            return

        # Save report to data/generated/ahp/ahp_report_sample.json
        out_path = self.reports_dir / "ahp_report_sample.json"
        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showwarning("Warning", f"Не удалось сохранить отчёт: {e}")

        # Display brief summary
        self._display_summary(report)

        # fill details panel but keep hidden
        self._fill_details(report)


AHPAnalysisTab = ConfigurationSelectionTab

__all__ = ["ConfigurationSelectionTab", "AHPAnalysisTab"]
