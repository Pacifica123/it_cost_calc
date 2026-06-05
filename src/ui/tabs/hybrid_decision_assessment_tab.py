"""Separate hybrid decision assessment panel for ПО/ТО workspaces."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import tkinter as tk
from tkinter import messagebox, ttk

from application.services.hybrid_decision_assessment_service import HybridDecisionAssessmentService
from ui.tabs.genetic_optimization_tab import GeneticOptimizationTab
from ui.widgets import attach_tooltip


_UNSET = object()


class HybridDecisionAssessmentTab(GeneticOptimizationTab):
    """Shows a scoped GA + AHP + Pareto-status summary without rerunning GA.

    The class inherits formatting helpers from ``GeneticOptimizationTab`` but
    deliberately replaces the content and action flow: the hybrid block reads the
    shared candidate pool and the latest standalone AHP/Pareto results pushed by
    the sibling panels in the same ПО/ТО workspace.
    """

    def _build_variables(self) -> None:
        super()._build_variables()
        self.hybrid_service = HybridDecisionAssessmentService()
        self.lambda_var = tk.StringVar(value="0.5")
        self.pool_status_var = tk.StringVar(value="Общий пул альтернатив ещё не передавался из GA.")
        self.ahp_status_var = tk.StringVar(value="AHP-результат ещё не получен.")
        self.pareto_status_var = tk.StringVar(value="Pareto-результат ещё не получен.")
        self.status_var.set("Гибридная оценка ещё не рассчитывалась.")
        self.explanation_var.set(
            "Гибридная панель не запускает GA и AHP заново. Она берёт общий пул кандидатов, "
            "последний самостоятельный AHP-результат и Pareto-статус из соседних блоков текущей области."
        )
        self.hybrid_summary_var.set("После расчёта здесь появится сводная рекомендация.")
        self._candidate_pool: list[Mapping[str, Any]] = []
        self._candidate_pool_source_label = "общий пул альтернатив"
        self._ahp_report: Mapping[str, Any] | None = None
        self._pareto_report: Mapping[str, Any] | None = None

    def _build_content(self) -> None:
        self.inner_frame.columnconfigure(0, weight=0, minsize=280)
        self.inner_frame.columnconfigure(1, weight=1)

        controls = ttk.LabelFrame(self.inner_frame, text="Гибрид: источники и режим")
        controls.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Область").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        fixed_label = self.profile_service.labels().get(self._analysis_scope(), self._analysis_scope())
        ttk.Label(controls, text=f"{fixed_label} (зафиксирована)").grid(
            row=0, column=1, sticky="w", padx=6, pady=4
        )
        ttk.Label(controls, text="λ к GA").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(controls, textvariable=self.lambda_var, width=8).grid(
            row=1, column=1, sticky="w", padx=6, pady=4
        )
        lambda_hint_text = (
            "λ=0.5 — нейтральный компромисс. Значение ближе к 1 усиливает GA, "
            "ближе к 0 усиливает AHP. Pareto остаётся статусом/предупреждением, а не скрытым весом."
        )
        lambda_hint = ttk.Label(controls, text="Как работает λ", justify="left")
        lambda_hint.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 8))
        attach_tooltip(lambda_hint, lambda_hint_text)

        status_box = ttk.LabelFrame(self.inner_frame, text="Состояние методов")
        status_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        for row, variable in enumerate(
            (self.pool_status_var, self.ahp_status_var, self.pareto_status_var, self.status_var)
        ):
            ttk.Label(status_box, textvariable=variable, wraplength=300, justify="left").grid(
                row=row, column=0, sticky="ew", padx=6, pady=4
            )

        actions = ttk.Frame(self.inner_frame)
        actions.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
        ttk.Button(
            actions,
            text="Обновить гибридную витрину",
            command=self.run_hybrid_assessment,
        ).pack(side="left")

        result_box = ttk.LabelFrame(self.inner_frame, text="Сводная витрина GA / AHP / Pareto / Hybrid")
        result_box.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")
        result_box.columnconfigure(0, weight=1)
        result_box.rowconfigure(1, weight=1)

        hybrid_result_hint = ttk.Label(
            result_box,
            text="Как читать гибрид ⓘ",
            justify="left",
            cursor="question_arrow",
        )
        hybrid_result_hint.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        attach_tooltip(hybrid_result_hint, self.explanation_var, wraplength=760)

        table_wrap = ttk.Frame(result_box)
        table_wrap.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        table_wrap.columnconfigure(0, weight=1)
        table_wrap.rowconfigure(0, weight=1)
        self.hybrid_ranking_table = ttk.Treeview(
            table_wrap,
            columns=(
                "rank",
                "name",
                "hybrid_score",
                "ga_rank",
                "ahp_rank",
                "pareto",
                "rank_disagreement",
                "comment",
            ),
            show="headings",
            height=14,
        )
        headings = {
            "rank": "Ранг",
            "name": "Альтернатива",
            "hybrid_score": "Hybrid score",
            "ga_rank": "GA",
            "ahp_rank": "AHP",
            "pareto": "Pareto",
            "rank_disagreement": "Δ ранга",
            "comment": "Комментарий",
        }
        for column, heading in headings.items():
            self.hybrid_ranking_table.heading(column, text=heading)
            width = 250 if column == "name" else 130
            if column in {"rank", "ga_rank", "ahp_rank"}:
                width = 70
            self.hybrid_ranking_table.column(column, width=width, anchor="w")
        scrollbar = ttk.Scrollbar(table_wrap, orient="vertical", command=self.hybrid_ranking_table.yview)
        self.hybrid_ranking_table.configure(yscrollcommand=scrollbar.set)
        self.hybrid_ranking_table.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        ttk.Label(
            result_box,
            textvariable=self.hybrid_summary_var,
            wraplength=760,
            justify="left",
        ).grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

        self.update_scrollregion()

    def load_candidate_pool(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        source_label: str = "общий пул альтернатив",
    ) -> None:
        self._candidate_pool = [dict(candidate) for candidate in candidates or []]
        self._candidate_pool_source_label = source_label
        self.last_result = None
        self._ahp_report = None
        self._pareto_report = None
        self.pool_status_var.set(f"Пул получен: {len(self._candidate_pool)} альтернатив; источник: {source_label}.")
        self.ahp_status_var.set("AHP-результат сброшен: пул изменился.")
        self.pareto_status_var.set("Pareto-результат сброшен: пул изменился.")
        self.status_var.set("Гибридная оценка ожидает самостоятельные результаты AHP/Pareto.")
        self.hybrid_summary_var.set("Пул обновлён; после AHP/Pareto нажмите обновление гибридной витрины.")
        self._clear_tree(self.hybrid_ranking_table)

    def clear_method_results(self) -> None:
        self.last_result = None
        self._ahp_report = None
        self._pareto_report = None
        self.ahp_status_var.set("AHP-результат сброшен: пул изменился.")
        self.pareto_status_var.set("Pareto-результат сброшен: пул изменился.")
        self.status_var.set("Гибридная оценка ожидает самостоятельные результаты AHP/Pareto.")
        self.hybrid_summary_var.set("Пул/методы изменились; предыдущая гибридная витрина очищена.")
        self._clear_tree(self.hybrid_ranking_table)

    def update_ahp_result(self, report: Mapping[str, Any] | None) -> None:
        self._ahp_report = report
        self.last_result = None
        self._clear_tree(self.hybrid_ranking_table)
        if report is None:
            self.ahp_status_var.set("AHP-результат ещё не получен.")
            return
        count = self._ahp_ranking_count(report)
        match_hint = self._pool_match_hint(report)
        self.ahp_status_var.set(f"AHP-результат получен: {count} ранжированных альтернатив.{match_hint}")
        self.status_var.set("AHP обновлён; нажмите обновление гибридной витрины.")

    def update_pareto_result(self, report: Mapping[str, Any] | None) -> None:
        self._pareto_report = report
        self.last_result = None
        self._clear_tree(self.hybrid_ranking_table)
        if report is None:
            self.pareto_status_var.set("Pareto-результат ещё не получен.")
            return
        count = len(report.get("ranking", [])) if isinstance(report.get("ranking"), list) else 0
        nondominated = len(report.get("final_nondominated", [])) if isinstance(report.get("final_nondominated"), list) else 0
        self.pareto_status_var.set(
            f"Pareto-результат получен: {count} альтернатив, недоминируемых: {nondominated}."
        )
        self.status_var.set("Pareto обновлён; нажмите обновление гибридной витрины.")

    def run_hybrid_assessment(self) -> None:
        try:
            lambda_value = float(self.lambda_var.get())
        except ValueError:
            messagebox.showerror("Ошибка ввода", "λ должен быть числом от 0 до 1.", parent=self)
            return

        assessment = self.hybrid_service.assess(
            self._candidate_pool,
            ahp_report=self._ahp_report,
            pareto_report=self._pareto_report,
            lambda_value=lambda_value,
            source_label=self._candidate_pool_source_label,
        )
        self.last_result = {"hybrid_assessment": assessment}
        self._render_assessment(assessment)

    def _render_assessment(self, assessment: Mapping[str, Any]) -> None:
        self._clear_tree(self.hybrid_ranking_table)
        if assessment.get("status") == "ok":
            self.status_var.set("Гибридная оценка рассчитана по общему пулу и последним результатам методов.")
            self.hybrid_summary_var.set(str(assessment.get("summary") or "Гибридная оценка рассчитана."))
        else:
            self.status_var.set(str(assessment.get("reason") or "Гибридная оценка не рассчитана."))
            warnings = assessment.get("warnings") if isinstance(assessment.get("warnings"), list) else []
            self.hybrid_summary_var.set(" ".join(str(item) for item in warnings) or self.status_var.get())

        for row in assessment.get("ranking", []) if isinstance(assessment.get("ranking"), list) else []:
            if not isinstance(row, Mapping):
                continue
            self.hybrid_ranking_table.insert(
                "",
                "end",
                values=(
                    row.get("rank", "—"),
                    row.get("name", "—"),
                    self._format_number(row.get("hybrid_score"), digits=4),
                    row.get("ga_rank", "—"),
                    row.get("ahp_rank", "—"),
                    row.get("pareto_status", "нет данных"),
                    self._format_number(row.get("rank_disagreement"), digits=0),
                    row.get("comment", ""),
                ),
            )
        self.update_scrollregion()

    def _ahp_ranking_count(self, report: Mapping[str, Any]) -> int:
        final = report.get("final") if isinstance(report.get("final"), Mapping) else {}
        ranking = final.get("ranking") if isinstance(final, Mapping) else []
        return len(ranking) if isinstance(ranking, list) else 0

    def _pool_match_hint(self, report: Mapping[str, Any]) -> str:
        pool_ids = {str(candidate.get("id")) for candidate in self._candidate_pool if candidate.get("id")}
        if not pool_ids:
            return " Пул пуст."
        final = report.get("final") if isinstance(report.get("final"), Mapping) else {}
        ranking = final.get("ranking") if isinstance(final, Mapping) else []
        ahp_ids: set[str] = set()
        if isinstance(ranking, list):
            for item in ranking:
                if isinstance(item, Mapping):
                    candidate_id = item.get("id") or item.get("alternative_id")
                elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and item:
                    candidate_id = item[0]
                else:
                    candidate_id = None
                if candidate_id:
                    ahp_ids.add(str(candidate_id))
        extra = ahp_ids - pool_ids
        missing = pool_ids - ahp_ids
        if not extra and not missing:
            return " ID совпадают с текущим пулом."
        return " Есть расхождение ID с текущим пулом; Hybrid лишние AHP-строки проигнорирует."


__all__ = ["HybridDecisionAssessmentTab"]
