"""Refactored criteria-importance UI tab assembled from smaller mixins."""

import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from .data_mixin import CriteriaImportanceDataMixin
from .tables_mixin import CriteriaImportanceTablesMixin
from .dialogs_mixin import CriteriaImportanceDialogsMixin
from .presenter_mixin import CriteriaImportancePresenterMixin

try:
    from it_cost_calc.domain.decision.criteria_importance.analysis import (
        load_default_budgeting_case,
        run_importance_pipeline,
        case_to_json,
    )
except Exception:
    load_default_budgeting_case = None
    run_importance_pipeline = None
    case_to_json = None


class CriteriaImportanceTab(
    CriteriaImportanceDialogsMixin,
    CriteriaImportanceTablesMixin,
    CriteriaImportancePresenterMixin,
    CriteriaImportanceDataMixin,
    tk.Frame,
):
    def __init__(self, parent, crud):
        super().__init__(parent)
        self.crud = crud
        self.case_data = None
        self.analysis_report = None
        self._build_ui()
        self._load_default_case()

    def _build_ui(self):
        toolbar = tk.Frame(self)
        toolbar.pack(fill="x", padx=6, pady=6)

        tk.Button(toolbar, text="Загрузить пример из лекции", command=self._load_default_case).pack(
            side="left"
        )
        tk.Button(toolbar, text="Загрузить JSON...", command=self._load_json).pack(side="left")
        tk.Button(toolbar, text="Сохранить JSON...", command=self._save_json).pack(side="left")
        tk.Button(toolbar, text="Рассчитать", command=self._run_analysis).pack(
            side="left", padx=(12, 0)
        )

        top_label = tk.Label(
            self,
            text=(
                "Модуль автоматизирует кейс выбора ИТ-решения по методике анализа важности критериев. "
                "В первой области можно вручную добавлять критерии как строки и системы как колонки."
            ),
            anchor="w",
            justify="left",
        )
        top_label.pack(fill="x", padx=6)

        main_pane = tk.PanedWindow(self, orient="horizontal", sashrelief="raised")
        main_pane.pack(fill="both", expand=True, padx=6, pady=6)

        left_panel = tk.Frame(main_pane)
        right_panel = tk.Frame(main_pane)
        main_pane.add(left_panel, stretch="always")
        main_pane.add(right_panel, stretch="always")

        scores_box = tk.LabelFrame(
            left_panel,
            text="1. Значения критериев для систем (двойной клик по оценке для редактирования)",
        )
        scores_box.pack(fill="both", expand=True, pady=(0, 6))

        scores_toolbar = tk.Frame(scores_box)
        scores_toolbar.pack(fill="x", padx=6, pady=(6, 2))
        tk.Button(scores_toolbar, text="Добавить критерий", command=self._add_criterion_popup).pack(
            side="left"
        )
        tk.Button(
            scores_toolbar, text="Изменить критерий", command=self._edit_selected_criterion
        ).pack(side="left")
        tk.Button(
            scores_toolbar, text="Удалить критерий", command=self._delete_selected_criterion
        ).pack(side="left")
        tk.Label(scores_toolbar, text="   ").pack(side="left")
        tk.Button(
            scores_toolbar, text="Добавить систему", command=self._add_alternative_popup
        ).pack(side="left")
        tk.Button(
            scores_toolbar, text="Изменить систему", command=self._edit_alternative_popup
        ).pack(side="left")
        tk.Button(
            scores_toolbar, text="Удалить систему", command=self._delete_alternative_popup
        ).pack(side="left")

        scores_hint = tk.Label(
            scores_box,
            text="Строка = критерий, колонка = система. ID и названия можно менять через кнопки сверху.",
            anchor="w",
            justify="left",
        )
        scores_hint.pack(fill="x", padx=6, pady=(0, 2))

        scores_tree_wrap = tk.Frame(scores_box)
        scores_tree_wrap.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.scores_tree = ttk.Treeview(scores_tree_wrap, show="headings", height=13)
        self.scores_tree.grid(row=0, column=0, sticky="nsew")
        self.scores_tree.bind("<Double-1>", self._on_score_double_click)

        scores_y = ttk.Scrollbar(
            scores_tree_wrap, orient="vertical", command=self.scores_tree.yview
        )
        scores_y.grid(row=0, column=1, sticky="ns")
        scores_x = ttk.Scrollbar(
            scores_tree_wrap, orient="horizontal", command=self.scores_tree.xview
        )
        scores_x.grid(row=1, column=0, sticky="ew")
        scores_tree_wrap.grid_rowconfigure(0, weight=1)
        scores_tree_wrap.grid_columnconfigure(0, weight=1)
        self.scores_tree.configure(yscrollcommand=scores_y.set, xscrollcommand=scores_x.set)

        relations_box = tk.LabelFrame(left_panel, text="2. Отношения важности критериев")
        relations_box.pack(fill="both", expand=True)

        relations_toolbar = tk.Frame(relations_box)
        relations_toolbar.pack(fill="x", padx=6, pady=(6, 2))
        tk.Button(relations_toolbar, text="Добавить связь", command=self._add_relation_popup).pack(
            side="left"
        )
        tk.Button(relations_toolbar, text="Изменить связь", command=self._edit_relation_popup).pack(
            side="left"
        )
        tk.Button(relations_toolbar, text="Удалить связь", command=self._delete_relation).pack(
            side="left"
        )

        self.relations_tree = ttk.Treeview(
            relations_box,
            columns=("left", "op", "factor", "right"),
            show="headings",
            height=10,
        )
        for col, title, width in [
            ("left", "Левый критерий", 130),
            ("op", "Операция", 80),
            ("factor", "Коэффициент", 100),
            ("right", "Правый критерий", 130),
        ]:
            self.relations_tree.heading(col, text=title)
            self.relations_tree.column(col, width=width, anchor="center")
        self.relations_tree.pack(fill="both", expand=True, padx=6, pady=(0, 6), side="left")
        self.relations_tree.bind("<Double-1>", self._edit_relation_popup)

        relations_scroll = ttk.Scrollbar(
            relations_box, orient="vertical", command=self.relations_tree.yview
        )
        relations_scroll.pack(fill="y", side="right", pady=(0, 6))
        self.relations_tree.configure(yscrollcommand=relations_scroll.set)

        result_box = tk.LabelFrame(right_panel, text="3. Результаты расчета")
        result_box.pack(fill="both", expand=True)

        self.result_notebook = ttk.Notebook(result_box)
        self.result_notebook.pack(fill="both", expand=True, padx=6, pady=6)

        summary_tab = tk.Frame(self.result_notebook)
        multiplicity_tab = tk.Frame(self.result_notebook)
        ranking_tab = tk.Frame(self.result_notebook)
        self.result_notebook.add(summary_tab, text="Сводка")
        self.result_notebook.add(multiplicity_tab, text="Кратности")
        self.result_notebook.add(ranking_tab, text="Ранжирование")

        self.result_text = tk.Text(summary_tab, wrap="word")
        self.result_text.pack(fill="both", expand=True, side="left")
        result_scroll = ttk.Scrollbar(
            summary_tab, orient="vertical", command=self.result_text.yview
        )
        result_scroll.pack(fill="y", side="right")
        self.result_text.configure(yscrollcommand=result_scroll.set)

        self.multiplicity_tree = ttk.Treeview(
            multiplicity_tab,
            columns=("criterion_id", "criterion_name", "multiplicity"),
            show="headings",
            height=14,
        )
        for col, title, width, anchor in [
            ("criterion_id", "ID", 80, "center"),
            ("criterion_name", "Критерий", 340, "w"),
            ("multiplicity", "Кратность", 100, "center"),
        ]:
            self.multiplicity_tree.heading(col, text=title)
            self.multiplicity_tree.column(col, width=width, anchor=anchor)
        self.multiplicity_tree.pack(fill="both", expand=True, side="left")
        mult_scroll = ttk.Scrollbar(
            multiplicity_tab, orient="vertical", command=self.multiplicity_tree.yview
        )
        mult_scroll.pack(fill="y", side="right")
        self.multiplicity_tree.configure(yscrollcommand=mult_scroll.set)

        self.ranking_tree = ttk.Treeview(
            ranking_tab,
            columns=("place", "name", "weighted_sum", "nondominated"),
            show="headings",
            height=14,
        )
        for col, title, width, anchor in [
            ("place", "Место", 70, "center"),
            ("name", "Система", 220, "w"),
            ("weighted_sum", "Взвешенная сумма", 150, "center"),
            ("nondominated", "Недоминируемая", 120, "center"),
        ]:
            self.ranking_tree.heading(col, text=title)
            self.ranking_tree.column(col, width=width, anchor=anchor)
        self.ranking_tree.pack(fill="both", expand=True, side="left")
        rank_scroll = ttk.Scrollbar(ranking_tab, orient="vertical", command=self.ranking_tree.yview)
        rank_scroll.pack(fill="y", side="right")
        self.ranking_tree.configure(yscrollcommand=rank_scroll.set)

    def _load_default_case(self):
        if load_default_budgeting_case is None:
            messagebox.showerror(
                "Ошибка",
                "Не удалось импортировать модуль domain.decision.criteria_importance.analysis",
            )
            return
        self.case_data = load_default_budgeting_case()
        self.analysis_report = None
        self._refresh_scores_tree()
        self._refresh_relations_tree()
        self._clear_result_tables()
        self._set_result(
            "Загружен демонстрационный пример выбора системы бюджетирования из лекции.\n"
            'Можно вручную добавлять критерии и системы, менять оценки и связи, затем нажать "Рассчитать".'
        )

    def _load_json(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self.case_data = json.load(fh)
            self.analysis_report = None
            self._refresh_scores_tree()
            self._refresh_relations_tree()
            self._clear_result_tables()
            self._set_result(f"Загружен JSON: {path}")
        except Exception as exc:
            messagebox.showerror("Ошибка загрузки", str(exc))

    def _save_json(self):
        if not self.case_data:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                if case_to_json is not None:
                    fh.write(case_to_json(self.case_data))
                else:
                    json.dump(self.case_data, fh, ensure_ascii=False, indent=2)
            self._set_result(f"Данные сохранены в {path}")
        except Exception as exc:
            messagebox.showerror("Ошибка сохранения", str(exc))

    def _run_analysis(self):
        if run_importance_pipeline is None:
            messagebox.showerror(
                "Ошибка",
                "Не удалось импортировать модуль domain.decision.criteria_importance.analysis",
            )
            return
        try:
            self.analysis_report = run_importance_pipeline(self.case_data)
            self._set_result(self._format_report(self.analysis_report))
            self._populate_result_tables(self.analysis_report)
            self.result_notebook.select(0)
        except Exception as exc:
            messagebox.showerror("Ошибка расчета", str(exc))
