"""User interface for running the genetic optimization scenario."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Mapping, Sequence

from application.use_cases.run_genetic_optimization import RunGeneticOptimizationUseCase
from application.use_cases.run_genetic_ahp_ranking import RunGeneticAhpRankingUseCase
from shared.validation import parse_float, parse_int
from ui.tabs.base_scrollable_tab import BaseScrollableTab


CATEGORY_LABELS = {
    "server": "Серверы",
    "client": "Клиенты",
    "network": "Сеть",
    "licenses": "Лицензии",
}

TERMINATION_LABELS = {
    "max_generations": "достигнут лимит поколений",
    "stagnation": "ранняя остановка по стагнации",
    "no_feasible_solution": "нет допустимого решения",
}


class GeneticOptimizationTab(BaseScrollableTab):
    """Runs GA over current runtime equipment and presents the best configuration.

    The tab intentionally describes IT-specific criteria and constraints outside
    the GA core. The universal kernel still receives only generic callable
    criteria, generic callable constraints and numeric weights.
    """

    def __init__(
        self,
        parent,
        run_genetic_optimization_use_case: RunGeneticOptimizationUseCase,
        *,
        run_genetic_ahp_ranking_use_case: RunGeneticAhpRankingUseCase | None = None,
        energy_tab: Any | None = None,
    ):
        super().__init__(parent, width=1180, height=620)
        self.run_genetic_optimization_use_case = run_genetic_optimization_use_case
        self.run_genetic_ahp_ranking_use_case = run_genetic_ahp_ranking_use_case
        self.energy_tab = energy_tab
        self.last_result: dict[str, Any] | None = None

        self._build_variables()
        self._build_content()

    def _build_variables(self) -> None:
        self.pop_size_var = tk.StringVar(value="40")
        self.generations_var = tk.StringVar(value="80")
        self.mutation_rate_var = tk.StringVar(value="0.04")
        self.seed_var = tk.StringVar(value="42")

        self.max_budget_var = tk.StringVar(value="600000")
        self.max_power_var = tk.StringVar(value="")
        self.target_users_var = tk.StringVar(value="3")

        self.coverage_weight_var = tk.StringVar(value="0.35")
        self.client_weight_var = tk.StringVar(value="0.25")
        self.power_weight_var = tk.StringVar(value="0.15")
        self.cost_weight_var = tk.StringVar(value="0.25")

        self.require_server_var = tk.BooleanVar(value=True)
        self.require_client_var = tk.BooleanVar(value=True)
        self.require_network_var = tk.BooleanVar(value=True)
        self.require_licenses_var = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="ГА ещё не запускался.")
        self.ahp_agreement_var = tk.StringVar(
            value="Согласованность ГА и AHP появится после запуска сценария «ГА + AHP»."
        )
        self.explanation_var = tk.StringVar(
            value=(
                "Задайте ограничения и веса критериев, затем нажмите «Запустить ГА». "
                "Ядро алгоритма получает универсальные критерии/ограничения и не зависит от ИТ-предметной области."
            )
        )

    def _build_content(self) -> None:
        self.inner_frame.columnconfigure(0, weight=1)
        self.inner_frame.columnconfigure(1, weight=2)

        settings = ttk.LabelFrame(self.inner_frame, text="Параметры запуска и ограничения")
        settings.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self._add_entry(settings, 0, "Размер популяции", self.pop_size_var)
        self._add_entry(settings, 1, "Количество поколений", self.generations_var)
        self._add_entry(settings, 2, "Вероятность мутации", self.mutation_rate_var)
        self._add_entry(settings, 3, "Seed", self.seed_var)
        ttk.Separator(settings, orient="horizontal").grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(8, 8)
        )
        self._add_entry(settings, 5, "Бюджет, руб.", self.max_budget_var)
        self._add_entry(settings, 6, "Максимальная мощность, Вт", self.max_power_var)
        self._add_entry(settings, 7, "Минимум пользователей/клиентских мест", self.target_users_var)

        required_box = ttk.LabelFrame(settings, text="Обязательные категории")
        required_box.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        for index, (text, variable) in enumerate(
            (
                ("Серверы", self.require_server_var),
                ("Клиенты", self.require_client_var),
                ("Сеть", self.require_network_var),
                ("Лицензии", self.require_licenses_var),
            )
        ):
            ttk.Checkbutton(required_box, text=text, variable=variable).grid(
                row=index // 2, column=index % 2, padx=6, pady=4, sticky="w"
            )

        weights = ttk.LabelFrame(self.inner_frame, text="Веса критериев")
        weights.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self._add_entry(weights, 0, "Покрытие категорий", self.coverage_weight_var)
        self._add_entry(weights, 1, "Клиентские места", self.client_weight_var)
        self._add_entry(weights, 2, "Энергопотребление", self.power_weight_var)
        self._add_entry(weights, 3, "Стоимость", self.cost_weight_var)
        ttk.Label(
            weights,
            text="Веса можно задавать не нормированными: сервис сам приведёт их к сумме 1.",
            wraplength=360,
            justify="left",
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=6, pady=(8, 0))

        actions = ttk.Frame(self.inner_frame)
        actions.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
        ttk.Button(actions, text="Запустить ГА", command=self.run_optimization).pack(side="left")
        ttk.Button(actions, text="ГА + AHP", command=self.run_ahp_ranking).pack(side="left", padx=(8, 0))
        ttk.Label(actions, textvariable=self.status_var, wraplength=360, justify="left").pack(
            side="left", padx=(12, 0), fill="x", expand=True
        )

        result_box = ttk.LabelFrame(self.inner_frame, text="Лучшее найденное решение")
        result_box.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")
        result_box.columnconfigure(0, weight=1)
        result_box.rowconfigure(1, weight=1)

        ttk.Label(
            result_box,
            textvariable=self.explanation_var,
            wraplength=680,
            justify="left",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        tables = ttk.Notebook(result_box)
        tables.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        composition_tab = ttk.Frame(tables)
        metrics_tab = ttk.Frame(tables)
        history_tab = ttk.Frame(tables)
        ahp_tab = ttk.Frame(tables)
        tables.add(composition_tab, text="Состав")
        tables.add(metrics_tab, text="Метрики")
        tables.add(history_tab, text="Сходимость")
        tables.add(ahp_tab, text="AHP-ранжирование")

        self.composition_table = self._build_tree(
            composition_tab,
            columns=("category", "name", "quantity", "cost", "power"),
            headings={
                "category": "Категория",
                "name": "Наименование",
                "quantity": "Кол-во",
                "cost": "Стоимость",
                "power": "Мощность, Вт",
            },
        )

        self.metrics_table = self._build_tree(
            metrics_tab,
            columns=("name", "raw", "normalized", "weight"),
            headings={
                "name": "Критерий/ограничение",
                "raw": "Значение",
                "normalized": "Норм. оценка",
                "weight": "Вес/статус",
            },
        )

        self.history_table = self._build_tree(
            history_tab,
            columns=("generation", "best", "generation_best", "mean", "feasible"),
            headings={
                "generation": "Поколение",
                "best": "Лучший score",
                "generation_best": "Score поколения",
                "mean": "Средний score",
                "feasible": "Допустимых",
            },
        )

        ttk.Label(
            ahp_tab,
            textvariable=self.ahp_agreement_var,
            wraplength=680,
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
                "ga_rank": "ГА-ранг",
                "ga_score": "ГА score",
                "rank_delta": "Δ ранга",
                "row_status": "Статус",
                "cost": "Стоимость",
            },
        )

        self.update_scrollregion()

    def _add_entry(
        self,
        parent: tk.Widget,
        row: int,
        label: str,
        variable: tk.StringVar,
        *,
        width: int = 16,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(parent, textvariable=variable, width=width).grid(
            row=row, column=1, sticky="ew", padx=6, pady=4
        )
        parent.columnconfigure(1, weight=1)

    def _build_tree(
        self,
        parent: tk.Widget,
        *,
        columns: Sequence[str],
        headings: Mapping[str, str],
    ) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=tuple(columns), show="headings", height=12)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=130 if column != "name" else 250, anchor="w")
        return tree

    def run_optimization(self) -> None:
        try:
            ga_params = self._read_ga_params()
            max_budget = self._optional_float(self.max_budget_var.get(), "Бюджет")
            max_power = self._optional_float(self.max_power_var.get(), "Максимальная мощность")
            target_users = self._optional_float(
                self.target_users_var.get(), "Минимум пользователей/клиентских мест"
            )
            weights = self._read_weights()
        except ValueError as error:
            messagebox.showerror("Ошибка ввода", str(error), parent=self)
            return

        power_lookup = self._power_lookup()
        criteria = self._build_criteria(power_lookup)
        constraints = self._build_constraints(power_lookup, max_power=max_power, target_users=target_users)
        required_categories = self._required_categories()

        try:
            result = self.run_genetic_optimization_use_case.execute(
                criteria=criteria,
                constraints=constraints,
                weights=weights,
                max_budget=max_budget,
                required_categories=required_categories,
                ga_params=ga_params,
            )
        except Exception as error:  # noqa: BLE001 - GUI должен показать пользователю причину отказа
            messagebox.showerror("Ошибка запуска ГА", str(error), parent=self)
            return

        self.last_result = result
        self._render_result(result, power_lookup)

    def run_ahp_ranking(self) -> None:
        if self.run_genetic_ahp_ranking_use_case is None:
            messagebox.showerror(
                "AHP недоступен",
                "Use case для связки ГА + AHP не настроен.",
                parent=self,
            )
            return

        try:
            ga_params = self._read_ga_params()
            ga_params["top_solutions_limit"] = 8
            max_budget = self._optional_float(self.max_budget_var.get(), "Бюджет")
            max_power = self._optional_float(self.max_power_var.get(), "Максимальная мощность")
            target_users = self._optional_float(
                self.target_users_var.get(), "Минимум пользователей/клиентских мест"
            )
            weights = self._read_weights()
        except ValueError as error:
            messagebox.showerror("Ошибка ввода", str(error), parent=self)
            return

        power_lookup = self._power_lookup()
        criteria = self._build_criteria(power_lookup)
        constraints = self._build_constraints(power_lookup, max_power=max_power, target_users=target_users)
        required_categories = self._required_categories()

        try:
            result = self.run_genetic_ahp_ranking_use_case.execute(
                criteria=criteria,
                constraints=constraints,
                weights=weights,
                ahp_weights=weights,
                ahp_top_limit=8,
                max_budget=max_budget,
                required_categories=required_categories,
                ga_params=ga_params,
            )
        except Exception as error:  # noqa: BLE001 - GUI должен показать пользователю причину отказа
            messagebox.showerror("Ошибка запуска ГА + AHP", str(error), parent=self)
            return

        self.last_result = result
        genetic_summary = result.get("genetic_optimization", {})
        if isinstance(genetic_summary, Mapping):
            self._render_result(genetic_summary, power_lookup)
        self._render_ahp_report(result.get("ahp_report", {}))

    def _read_ga_params(self) -> dict[str, Any]:
        seed_value = self.seed_var.get().strip()
        params: dict[str, Any] = {
            "pop_size": parse_int(self.pop_size_var.get(), "Размер популяции"),
            "generations": parse_int(self.generations_var.get(), "Количество поколений"),
            "mutation_rate": parse_float(self.mutation_rate_var.get(), "Вероятность мутации"),
            "init_mode": "heuristic_init",
        }
        if seed_value:
            params["seed"] = parse_int(seed_value, "Seed")
        return params

    def _read_weights(self) -> list[float]:
        return [
            parse_float(self.coverage_weight_var.get(), "Вес покрытия категорий"),
            parse_float(self.client_weight_var.get(), "Вес клиентских мест"),
            parse_float(self.power_weight_var.get(), "Вес энергопотребления"),
            parse_float(self.cost_weight_var.get(), "Вес стоимости"),
        ]

    def _optional_float(self, value: str, field_name: str) -> float | None:
        if not value.strip():
            return None
        return parse_float(value, field_name)

    def _required_categories(self) -> list[str]:
        result = []
        if self.require_server_var.get():
            result.append("server")
        if self.require_client_var.get():
            result.append("client")
        if self.require_network_var.get():
            result.append("network")
        if self.require_licenses_var.get():
            result.append("licenses")
        return result

    def _power_lookup(self) -> dict[str, float]:
        if self.energy_tab is not None and hasattr(self.energy_tab, "update_equipment_table"):
            self.energy_tab.update_equipment_table()

        equipment_data = getattr(self.energy_tab, "equipment_data", {}) or {}
        lookup: dict[str, float] = {}
        for name, row in equipment_data.items():
            try:
                lookup[str(name)] = float(row.get("max_power", 0.0))
            except (TypeError, ValueError, AttributeError):
                lookup[str(name)] = 0.0
        return lookup

    def _build_criteria(self, power_lookup: Mapping[str, float]) -> list[dict[str, Any]]:
        return [
            {
                "name": "category_coverage",
                "func": self._category_coverage,
                "direction": "max",
            },
            {
                "name": "client_capacity",
                "func": self._client_capacity,
                "direction": "max",
            },
            {
                "name": "total_power_watts",
                "func": lambda subset, lookup=power_lookup: self._total_power(subset, lookup),
                "direction": "min",
            },
            {
                "name": "capital_cost",
                "func": lambda subset: self._sum_property(subset, "total_cost"),
                "direction": "min",
            },
        ]

    def _build_constraints(
        self,
        power_lookup: Mapping[str, float],
        *,
        max_power: float | None,
        target_users: float | None,
    ) -> list[dict[str, Any]]:
        constraints: list[dict[str, Any]] = []
        if max_power is not None:
            constraints.append(
                {
                    "name": "power_limit",
                    "func": lambda subset, lookup=power_lookup: self._total_power(subset, lookup),
                    "operator": "<=",
                    "bound": max_power,
                }
            )
        if target_users is not None and target_users > 0:
            constraints.append(
                {
                    "name": "client_capacity_required",
                    "func": self._client_capacity,
                    "operator": ">=",
                    "bound": target_users,
                }
            )
        return constraints

    def _render_result(self, result: Mapping[str, Any], power_lookup: Mapping[str, float]) -> None:
        self._clear_tree(self.composition_table)
        self._clear_tree(self.metrics_table)
        self._clear_tree(self.history_table)
        if hasattr(self, "ahp_ranking_table"):
            self._clear_tree(self.ahp_ranking_table)
        if hasattr(self, "ahp_agreement_var"):
            self.ahp_agreement_var.set(
                "Согласованность ГА и AHP появится после запуска сценария «ГА + AHP»."
            )

        ga_result = result.get("ga_result", {}) if isinstance(result.get("ga_result"), Mapping) else {}
        if result.get("status") == "error":
            error = result.get("error") or ga_result.get("error") or "неизвестная ошибка"
            self.status_var.set("ГА завершился без допустимого решения.")
            self.explanation_var.set(f"Не удалось подобрать конфигурацию: {error}")
            return

        selected_items = list(result.get("selected_items", []))
        for item in selected_items:
            name = str(item.get("name", ""))
            quantity = self._number(item.get("quantity"), default=0.0)
            cost = self._number(item.get("total_cost"), default=0.0)
            power = self._item_power(item, power_lookup)
            category = str(item.get("source_category") or item.get("category") or "unknown")
            self.composition_table.insert(
                "",
                "end",
                values=(
                    CATEGORY_LABELS.get(category, category),
                    name,
                    self._format_number(quantity),
                    self._format_money(cost),
                    self._format_number(power),
                ),
            )

        for criterion in result.get("criteria", []):
            name = str(criterion.get("name", ""))
            self.metrics_table.insert(
                "",
                "end",
                values=(
                    self._criterion_label(name),
                    self._format_number(criterion.get("raw_value")),
                    self._format_number(criterion.get("normalized_value"), digits=3),
                    self._format_number(criterion.get("weight"), digits=3),
                ),
            )

        for constraint in result.get("constraints", []):
            status = "выполнено" if constraint.get("passed") else "нарушено"
            label = self._constraint_label(str(constraint.get("name", "")))
            value = self._format_number(constraint.get("value"))
            bound = self._format_number(constraint.get("bound"))
            operator_text = constraint.get("operator") or ""
            self.metrics_table.insert(
                "",
                "end",
                values=(label, f"{value} {operator_text} {bound}".strip(), "—", status),
            )

        history = list(ga_result.get("history", []))
        for row in self._history_sample(history):
            self.history_table.insert(
                "",
                "end",
                values=(
                    row.get("generation"),
                    self._format_number(row.get("best_score"), digits=4),
                    self._format_number(row.get("generation_best_score"), digits=4),
                    self._format_number(row.get("mean_score"), digits=4),
                    row.get("feasible_count"),
                ),
            )

        totals = result.get("totals", {}) if isinstance(result.get("totals"), Mapping) else {}
        score = ga_result.get("best_agg")
        termination = TERMINATION_LABELS.get(
            str(ga_result.get("termination_reason")), str(ga_result.get("termination_reason"))
        )
        self.status_var.set(
            f"Найдено элементов: {totals.get('selected_count', 0)}; score={self._format_number(score, digits=4)}."
        )
        self.explanation_var.set(
            "Выбрана конфигурация с максимальной нормализованной оценкой по заданным весам. "
            f"Итоговая стоимость: {self._format_money(totals.get('capital_cost'))}; "
            f"клиентских мест: {self._format_number(self._client_capacity_from_dicts(selected_items))}; "
            f"мощность: {self._format_number(sum(self._item_power(item, power_lookup) for item in selected_items))} Вт; "
            f"причина остановки: {termination}."
        )
        self.update_scrollregion()

    def _render_ahp_report(self, report: Any) -> None:
        if not hasattr(self, "ahp_ranking_table"):
            return
        self._clear_tree(self.ahp_ranking_table)
        if not isinstance(report, Mapping):
            return
        if report.get("status") != "ok":
            reason = report.get("reason") or "AHP-ранжирование не выполнено."
            self.status_var.set(str(reason))
            self.ahp_agreement_var.set(str(reason))
            return

        agreement = report.get("agreement", {}) if isinstance(report.get("agreement"), Mapping) else {}
        rank_delta_by_id = self._rank_delta_by_id(agreement)
        self.ahp_agreement_var.set(self._agreement_text(agreement))

        ranking = report.get("final", {}).get("ranking", []) if isinstance(report.get("final"), Mapping) else []
        for row in ranking:
            totals = row.get("totals", {}) if isinstance(row.get("totals"), Mapping) else {}
            delta = rank_delta_by_id.get(
                str(row.get("id")),
                self._rank_delta_from_row(row),
            )
            self.ahp_ranking_table.insert(
                "",
                "end",
                values=(
                    row.get("rank"),
                    row.get("name"),
                    self._format_number(row.get("score"), digits=4),
                    row.get("ga_rank"),
                    self._format_number(row.get("ga_score"), digits=4),
                    self._format_rank_delta(delta),
                    self._rank_delta_status(delta),
                    self._format_money(totals.get("capital_cost")),
                ),
            )

        winner = report.get("final", {}).get("winner", {}) if isinstance(report.get("final"), Mapping) else {}
        if winner:
            agreement_status = str(agreement.get("status_label") or "согласованность не рассчитана")
            self.status_var.set(
                f"AHP выбрал: {winner.get('name')} со score={self._format_number(winner.get('score'), digits=4)}; "
                f"ГА/AHP: {agreement_status}."
            )
            self.explanation_var.set(
                "ГА сформировал пул допустимых конфигураций, после чего AHP независимо оценил те же альтернативы. "
                f"{self._agreement_text(agreement)}"
            )
        self.update_scrollregion()

    def _rank_delta_by_id(self, agreement: Mapping[str, Any]) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in agreement.get("rank_deltas", []) if isinstance(agreement.get("rank_deltas"), list) else []:
            if not isinstance(row, Mapping):
                continue
            candidate_id = str(row.get("id") or "")
            if not candidate_id:
                continue
            try:
                result[candidate_id] = int(row.get("rank_delta"))
            except (TypeError, ValueError):
                continue
        return result

    def _rank_delta_from_row(self, row: Mapping[str, Any]) -> int | None:
        try:
            ahp_rank = int(row.get("rank"))
            ga_rank = int(row.get("ga_rank"))
        except (TypeError, ValueError):
            return None
        return ahp_rank - ga_rank

    def _format_rank_delta(self, delta: int | None) -> str:
        if delta is None:
            return "—"
        if delta == 0:
            return "0"
        return f"{delta:+d}"

    def _rank_delta_status(self, delta: int | None) -> str:
        if delta is None:
            return "нет данных"
        if delta == 0:
            return "совпадает"
        if abs(delta) <= 1:
            return "детальная перестановка"
        if abs(delta) <= 3:
            return "заметный сдвиг"
        return "сильное расхождение"

    def _agreement_text(self, agreement: Mapping[str, Any]) -> str:
        if not agreement:
            return "Согласованность ГА и AHP не рассчитана."
        summary = str(agreement.get("summary") or "").strip()
        warnings = agreement.get("warnings", [])
        warning_text = ""
        if isinstance(warnings, list) and warnings:
            warning_text = " " + " ".join(str(item) for item in warnings[:2])
        return (summary or "Согласованность ГА и AHP рассчитана.") + warning_text

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        tree.delete(*tree.get_children())

    def _history_sample(self, history: Sequence[Mapping[str, Any]], limit: int = 12) -> list[Mapping[str, Any]]:
        if len(history) <= limit:
            return list(history)
        head_count = max(1, limit // 3)
        tail_count = limit - head_count
        return list(history[:head_count]) + list(history[-tail_count:])

    def _category_coverage(self, subset: Sequence[Any]) -> float:
        return float(
            len(
                {
                    str(item.properties.get("source_category") or item.properties.get("category"))
                    for item in subset
                    if item.properties.get("source_category") or item.properties.get("category")
                }
            )
        )

    def _client_capacity(self, subset: Sequence[Any]) -> float:
        return sum(
            self._number(item.properties.get("quantity"), default=0.0)
            for item in subset
            if item.properties.get("source_category") == "client"
            or item.properties.get("category") == "client"
        )

    def _client_capacity_from_dicts(self, selected_items: Sequence[Mapping[str, Any]]) -> float:
        return sum(
            self._number(item.get("quantity"), default=0.0)
            for item in selected_items
            if item.get("source_category") == "client" or item.get("category") == "client"
        )

    def _total_power(self, subset: Sequence[Any], power_lookup: Mapping[str, float]) -> float:
        return sum(self._item_power(item.properties, power_lookup) for item in subset)

    def _item_power(self, item: Mapping[str, Any], power_lookup: Mapping[str, float]) -> float:
        name = str(item.get("name", ""))
        quantity = self._number(item.get("quantity"), default=1.0)
        per_unit_power = self._number(item.get("max_power", power_lookup.get(name, 0.0)), default=0.0)
        return quantity * per_unit_power

    def _sum_property(self, subset: Sequence[Any], property_name: str) -> float:
        return sum(self._number(item.properties.get(property_name), default=0.0) for item in subset)

    def _number(self, value: Any, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _format_number(self, value: Any, *, digits: int = 2) -> str:
        if value is None:
            return "—"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if abs(numeric - round(numeric)) < 10 ** (-(digits + 1)):
            return str(int(round(numeric)))
        return f"{numeric:.{digits}f}"

    def _format_money(self, value: Any) -> str:
        return f"{self._format_number(value, digits=2)} руб."

    def _criterion_label(self, name: str) -> str:
        return {
            "category_coverage": "Покрытие категорий",
            "client_capacity": "Клиентские места",
            "total_power_watts": "Энергопотребление",
            "capital_cost": "Стоимость",
        }.get(name, name)

    def _constraint_label(self, name: str) -> str:
        if name.startswith("required_category_"):
            category = name.removeprefix("required_category_")
            return f"Обязательная категория: {CATEGORY_LABELS.get(category, category)}"
        return {
            "budget": "Бюджет",
            "min_selected_items": "Минимум выбранных элементов",
            "max_selected_items": "Максимум выбранных элементов",
            "power_limit": "Лимит мощности",
            "client_capacity_required": "Минимум клиентских мест",
        }.get(name, name)


__all__ = ["GeneticOptimizationTab"]
