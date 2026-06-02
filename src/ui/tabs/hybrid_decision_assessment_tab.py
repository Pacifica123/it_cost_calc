"""Separate hybrid decision assessment panel for ПО/ТО workspaces."""

from __future__ import annotations

from typing import Any, Mapping

import tkinter as tk
from tkinter import ttk

from shared.constants import ANALYSIS_SCOPE_SOFTWARE
from ui.tabs.genetic_optimization_tab import GeneticOptimizationTab


class HybridDecisionAssessmentTab(GeneticOptimizationTab):
    """Runs the existing GA→AHP orchestration as a separate hybrid panel.

    The implementation deliberately reuses the already tested application use
    case, but removes this scenario from the visual GA block.  In the UI it is a
    fourth decision method panel: GA searches candidates, AHP has its own panel,
    Pareto/criteria keeps its own panel, and this tab only presents the combined
    consistency/hybrid view.
    """

    def _build_variables(self) -> None:
        super()._build_variables()
        self.status_var.set("Гибридная оценка ещё не рассчитывалась.")
        self.explanation_var.set(
            "Гибридная панель собирает GA-кандидатов, независимый AHP-рейтинг и итоговую "
            "сводную оценку в одном месте. Чистый результат GA теперь находится в отдельном блоке."
        )
        self.ahp_agreement_var.set(
            "После расчёта здесь появится сравнение лидеров GA и AHP по одному пулу кандидатов."
        )
        self.hybrid_summary_var.set(
            "После расчёта здесь появится итоговый гибридный лидер и предупреждения о расхождениях."
        )

    def _build_content(self) -> None:
        self.inner_frame.columnconfigure(0, weight=0, minsize=280)
        self.inner_frame.columnconfigure(1, weight=1)

        settings = ttk.LabelFrame(self.inner_frame, text="Гибрид: входные ограничения")
        settings.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        settings.columnconfigure(0, weight=0, minsize=132)
        settings.columnconfigure(1, weight=1)

        self._add_entry(settings, 0, "Популяция", self.pop_size_var)
        self._add_entry(settings, 1, "Поколений", self.generations_var)
        self._add_entry(settings, 2, "Мутация", self.mutation_rate_var)
        self._add_entry(settings, 3, "Seed", self.seed_var)
        ttk.Separator(settings, orient="horizontal").grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(8, 8)
        )
        self._add_entry(settings, 5, "Бюджет, руб.", self.max_budget_var)
        self._add_entry(settings, 6, "Макс. мощность, Вт", self.max_power_var)
        self._add_entry(settings, 7, "Мин. мест/лицензий", self.target_users_var)

        category_box = ttk.LabelFrame(settings, text="Фильтр категорий: + / − / пусто")
        category_box.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self._build_category_policy_controls(category_box)

        scope_box = ttk.LabelFrame(settings, text="Область анализа")
        scope_box.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        fixed_label = self.profile_service.labels().get(self._analysis_scope(), self._analysis_scope())
        ttk.Label(
            scope_box,
            text=f"Зафиксировано внутри текущей вкладки: {fixed_label}",
            wraplength=280,
            justify="left",
        ).grid(row=0, column=0, padx=6, pady=4, sticky="w")
        ttk.Label(
            settings,
            textvariable=self.scope_hint_var,
            wraplength=300,
            justify="left",
        ).grid(row=10, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        weights = ttk.LabelFrame(self.inner_frame, text="Мягкие веса для расчёта")
        weights.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        weights.columnconfigure(0, weight=0, minsize=132)
        weights.columnconfigure(1, weight=1)
        self._add_entry(weights, 0, "Энергия", self.power_weight_var)
        self._add_entry(weights, 1, "Стоимость", self.cost_weight_var)
        ttk.Label(
            weights,
            text=(
                "Минимум рабочих мест/лицензий и политика категорий являются фильтрами. "
                "Оставшиеся веса используются для GA и AHP-сравнения одного пула кандидатов."
            ),
            wraplength=300,
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(8, 0))

        actions = ttk.Frame(self.inner_frame)
        actions.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
        actions.columnconfigure(0, weight=1)
        ttk.Button(
            actions,
            text="Рассчитать гибридную оценку",
            command=self.run_ahp_ranking,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(actions, textvariable=self.status_var, wraplength=260, justify="left").grid(
            row=1, column=0, sticky="ew", pady=(6, 0)
        )

        result_box = ttk.LabelFrame(self.inner_frame, text="Сводная витрина GA / AHP / Hybrid")
        result_box.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")
        result_box.columnconfigure(0, weight=1)
        result_box.rowconfigure(1, weight=1)

        ttk.Label(
            result_box,
            textvariable=self.explanation_var,
            wraplength=720,
            justify="left",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        tables = ttk.Notebook(result_box)
        tables.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        ahp_tab = ttk.Frame(tables)
        hybrid_tab = ttk.Frame(tables)
        tables.add(ahp_tab, text="AHP по GA-кандидатам")
        tables.add(hybrid_tab, text="Гибридный рейтинг")

        ttk.Label(
            ahp_tab,
            textvariable=self.ahp_agreement_var,
            wraplength=720,
            justify="left",
        ).pack(fill="x", padx=4, pady=(4, 8))
        self.ahp_ranking_table = self._build_tree(
            ahp_tab,
            columns=(
                "rank",
                "name",
                "ahp_score",
                "ga_rank",
                "ga_score",
                "rank_delta",
                "row_status",
                "cost",
            ),
            headings={
                "rank": "Место AHP",
                "name": "Кандидатная конфигурация",
                "ahp_score": "AHP score",
                "ga_rank": "GA-ранг",
                "ga_score": "GA score",
                "rank_delta": "Δ ранга",
                "row_status": "Статус",
                "cost": "Стоимость",
            },
        )

        ttk.Label(
            hybrid_tab,
            textvariable=self.hybrid_summary_var,
            wraplength=720,
            justify="left",
        ).pack(fill="x", padx=4, pady=(4, 8))
        self.hybrid_ranking_table = self._build_tree(
            hybrid_tab,
            columns=(
                "rank",
                "name",
                "hybrid_score",
                "ga_rank",
                "ahp_rank",
                "score_disagreement",
                "comment",
            ),
            headings={
                "rank": "Гибридный ранг",
                "name": "Кандидатная конфигурация",
                "hybrid_score": "Hybrid score",
                "ga_rank": "GA-ранг",
                "ahp_rank": "AHP-ранг",
                "score_disagreement": "Расхождение",
                "comment": "Комментарий",
            },
        )

        self._sync_scope_hint()
        self.update_scrollregion()

    def _render_result(self, result: Mapping[str, Any], power_lookup: Mapping[str, float]) -> None:
        """Keep GA details out of the hybrid panel and show only pool status."""

        if result.get("status") == "error":
            self.status_var.set("GA не сформировал пул для гибридной оценки.")
            self.explanation_var.set(str(result.get("error") or "Нет допустимых кандидатов."))
            return

        candidate_count = len(result.get("candidate_solutions", []))
        totals = result.get("totals", {}) if isinstance(result.get("totals"), Mapping) else {}
        scope = getattr(self, "_last_analysis_scope", self._analysis_scope())
        unit_label = "лицензий" if scope == ANALYSIS_SCOPE_SOFTWARE else "клиентских мест"
        units = (
            self._software_quantity_from_dicts(result.get("selected_items", []))
            if scope == ANALYSIS_SCOPE_SOFTWARE
            else self._client_capacity_from_dicts(result.get("selected_items", []))
        )
        self.status_var.set(
            f"Пул GA-кандидатов: {candidate_count}; стоимость лучшего GA: "
            f"{self._format_money(totals.get('capital_cost'))}; {unit_label}: {self._format_number(units)}."
        )


__all__ = ["HybridDecisionAssessmentTab"]
