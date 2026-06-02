"""Composite ПО/ТО workspace tabs with collapsible and resizable panels."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Mapping

from application.services.analysis_scope_profile_service import AnalysisScopeProfileService
from application.services.equipment_service import EquipmentService
from application.services.scoped_candidate_pool_service import ScopedCandidatePoolService
from application.use_cases.run_genetic_ahp_ranking import RunGeneticAhpRankingUseCase
from application.use_cases.run_genetic_optimization import RunGeneticOptimizationUseCase
from shared.constants import ANALYSIS_SCOPE_SOFTWARE, ANALYSIS_SCOPE_TECHNICAL
from ui.tabs.capex_tab import CapexTab, SoftwareTab, TechnicalEquipmentTab
from ui.tabs.configuration_selection_tab import ConfigurationSelectionTab
from ui.tabs.criteria_importance_tab import CriteriaImportanceTab
from ui.tabs.genetic_optimization_tab import GeneticOptimizationTab
from ui.tabs.hybrid_decision_assessment_tab import HybridDecisionAssessmentTab
from ui.tabs.opex_tab import OpexTab
from ui.theme import SUBTLE_BUTTON, SUBTLE_BUTTON_HOVER, TEXT, WORKSPACE, force_panel_backgrounds
from ui.widgets import CollapsiblePanel


TECHNICAL_OPEX_CONFIG = {
    "server_rental": ("Аренда серверов", ("name", "monthly_cost")),
    "backup": ("Резервирование", ("name", "monthly_cost")),
    "labor_costs": ("Оплата труда", ("name", "monthly_cost")),
    "server_administration": ("Администрирование серверов", ("name", "monthly_cost")),
}

SOFTWARE_OPEX_CONFIG = {
    "subscription_licenses": ("Лицензии по подписке", ("name", "monthly_cost")),
    "migration": ("Миграция и внедрение", ("name", "one_time_cost")),
    "testing": ("Тестирование и приёмка", ("name", "one_time_cost")),
}

COST_RELATION_ROWS = (
    (
        "Покупка серверов, ПК, сетевого оборудования",
        "ТО",
        "CAPEX",
        "Формирует начальные инвестиции и участвует в расчёте мощности/клиентских мест.",
    ),
    (
        "Аренда серверов и регулярное администрирование",
        "ТО",
        "OPEX",
        "Периодические платежи не являются покупкой актива и влияют на TCO/NPV как текущие расходы.",
    ),
    (
        "Бессрочные лицензии и разовая покупка ПО",
        "ПО",
        "CAPEX",
        "По модели проекта учитывается как разовое приобретение программного актива.",
    ),
    (
        "Подписки SaaS/лицензии на месяц или год",
        "ПО",
        "OPEX",
        "Повторяются во времени, поэтому должны попадать в периодические расходы.",
    ),
    (
        "Миграция, тестирование, внедрение",
        "ПО/ТО",
        "OPEX, разовые",
        "Это не покупка оборудования, а проектные работы; в текущей модели хранятся как единовременный OPEX.",
    ),
    (
        "Электроэнергия и эксплуатация",
        "ТО",
        "OPEX",
        "Зависит от технической конфигурации и добавляется в финансовую модель отдельно.",
    ),
)


class SolutionAreaWorkspaceTab(tk.Frame):
    """Workspace for one analysis scope: source data, costs and decision methods."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        scope: str,
        equipment_service: EquipmentService,
        crud: Any,
        run_genetic_optimization_use_case: RunGeneticOptimizationUseCase,
        run_genetic_ahp_ranking_use_case: RunGeneticAhpRankingUseCase,
        energy_tab: Any | None,
        data_root: Any | None,
        profile_service: AnalysisScopeProfileService,
    ) -> None:
        super().__init__(parent, background=WORKSPACE)
        self.scope = scope
        self.equipment_service = equipment_service
        self.crud = crud
        self.profile_service = profile_service
        self.candidate_pool_service = ScopedCandidatePoolService(profile_service=self.profile_service)
        self.decision_state: dict[str, Any] = {
            "candidate_pool_snapshot": None,
            "ahp_result": None,
            "pareto_result": None,
            "hybrid_result": None,
        }
        self.panels: dict[str, CollapsiblePanel] = {}
        self._adaptive_pane_groups: list[dict[str, Any]] = []
        self._adaptive_after_ids: list[str] = []
        self.bind("<Destroy>", self._cancel_adaptive_rebalance_callbacks, add="+")

        self._build_layout(
            run_genetic_optimization_use_case=run_genetic_optimization_use_case,
            run_genetic_ahp_ranking_use_case=run_genetic_ahp_ranking_use_case,
            energy_tab=energy_tab,
            data_root=data_root,
        )

    @property
    def scope_label(self) -> str:
        return self.profile_service.labels().get(self.scope, self.scope)

    @property
    def scope_title(self) -> str:
        return self.profile_service.get_profile(self.scope).title

    def _build_layout(
        self,
        *,
        run_genetic_optimization_use_case: RunGeneticOptimizationUseCase,
        run_genetic_ahp_ranking_use_case: RunGeneticAhpRankingUseCase,
        energy_tab: Any | None,
        data_root: Any | None,
    ) -> None:
        header = tk.Frame(self, background=WORKSPACE, padx=12, pady=10)
        header.pack(fill="x")
        header.columnconfigure(1, weight=1)
        tk.Label(
            header,
            text=self.scope_title,
            font=("TkDefaultFont", 12, "bold"),
            background=WORKSPACE,
            foreground=TEXT,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header,
            text=self._workspace_hint(),
            wraplength=720,
            justify="left",
            background=WORKSPACE,
            foreground=TEXT,
        ).grid(row=0, column=1, padx=(16, 10), sticky="ew")
        self._build_header_button(header, "Свернуть анализ", self.close_analysis).grid(
            row=0, column=2, padx=(0, 6), sticky="e"
        )
        self._build_header_button(header, "Раскрыть всё", self.open_all).grid(
            row=0, column=3, sticky="e"
        )

        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        data_side = ttk.Panedwindow(main, orient="vertical")
        analysis_side = ttk.Panedwindow(main, orient="vertical")
        main.add(data_side, weight=4)
        main.add(analysis_side, weight=8)

        self.after_idle(lambda: self._set_initial_pane_positions(main, data_side, analysis_side))

        self._add_cost_input_panels(data_side)
        self._add_analysis_panels(
            analysis_side,
            run_genetic_optimization_use_case=run_genetic_optimization_use_case,
            run_genetic_ahp_ranking_use_case=run_genetic_ahp_ranking_use_case,
            energy_tab=energy_tab,
            data_root=data_root,
        )
        self.after_idle(self._force_panel_backgrounds)
        self._schedule_adaptive_after_idle(self._rebalance_all_adaptive_panes)
        self._schedule_adaptive_after(120, self._rebalance_all_adaptive_panes)

    def _build_header_button(self, parent: tk.Misc, text: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            background=SUBTLE_BUTTON,
            activebackground=SUBTLE_BUTTON_HOVER,
            foreground=TEXT,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=10,
            pady=6,
            cursor="hand2",
        )

    def _workspace_hint(self) -> str:
        if self.scope == ANALYSIS_SCOPE_SOFTWARE:
            return "Лицензии, подписки, GA/AHP/Pareto и финансовая классификация ПО в одном рабочем пространстве."
        return "Оборудование, эксплуатация, GA/AHP/Pareto и связь ТО с CAPEX/OPEX в одном рабочем пространстве."

    def _set_initial_pane_positions(
        self,
        main: ttk.Panedwindow,
        data_side: ttk.Panedwindow,
        analysis_side: ttk.Panedwindow,
    ) -> None:
        """Give dense panes usable first sizes without taking resizing away from the user."""

        try:
            width = max(main.winfo_width(), 1)
            height = max(self.winfo_height(), 1)
            if width > 900:
                main.sashpos(0, int(width * 0.36))
            if height > 520:
                data_side.sashpos(0, int(height * 0.44))
                data_side.sashpos(1, int(height * 0.74))
                analysis_side.sashpos(0, int(height * 0.72))
                analysis_side.sashpos(1, int(height * 0.82))
                analysis_side.sashpos(2, int(height * 0.91))
            self._schedule_rebalance_all_adaptive_panes()
        except tk.TclError:
            return

    def _add_cost_input_panels(self, parent: ttk.Panedwindow) -> None:
        capex_panel = self._panel(
            parent,
            key="capex",
            title=f"{self.scope_label}: капитальные позиции",
            hint="разовые активы",
        )
        if self.scope == ANALYSIS_SCOPE_SOFTWARE:
            self.capex_tab: CapexTab = SoftwareTab(
                capex_panel.content,
                self.equipment_service,
                columns_count=1,
                compact_tables=True,
            )
        else:
            self.capex_tab = TechnicalEquipmentTab(
                capex_panel.content,
                self.equipment_service,
                columns_count=1,
                compact_tables=True,
            )
        self.capex_tab.pack(fill="both", expand=True)
        self._add_adaptive_panel(parent, capex_panel, weight=1)

        opex_panel = self._panel(
            parent,
            key="opex",
            title=f"{self.scope_label}: операционные и проектные затраты",
            hint="подписки, аренда, внедрение",
        )
        if self.scope == ANALYSIS_SCOPE_SOFTWARE:
            config: Mapping[str, tuple[str, tuple[str, ...]]] = SOFTWARE_OPEX_CONFIG
            intro = (
                "ПО/OPEX: подписочные лицензии учитываются как периодические расходы, "
                "а миграция и тестирование — как разовые проектные расходы внедрения."
            )
        else:
            config = TECHNICAL_OPEX_CONFIG
            intro = (
                "ТО/OPEX: аренда, резервирование, администрирование и трудозатраты "
                "описывают эксплуатацию технической инфраструктуры без покупки актива."
            )
        self.opex_tab = OpexTab(
            opex_panel.content,
            self.equipment_service,
            entity_config=config,
            intro_text=intro,
            columns_count=1,
            compact_tables=True,
        )
        self.opex_tab.pack(fill="both", expand=True)
        self._add_adaptive_panel(parent, opex_panel, weight=1)

        relation_panel = self._panel(
            parent,
            key="cost_relation",
            title="Матрица ПО/ТО × CAPEX/OPEX",
            hint="финансовая классификация",
            initially_open=False,
        )
        self._build_cost_relation_table(relation_panel.content)
        self._add_adaptive_panel(parent, relation_panel, weight=1)

    def _add_analysis_panels(
        self,
        parent: ttk.Panedwindow,
        *,
        run_genetic_optimization_use_case: RunGeneticOptimizationUseCase,
        run_genetic_ahp_ranking_use_case: RunGeneticAhpRankingUseCase,
        energy_tab: Any | None,
        data_root: Any | None,
    ) -> None:
        ga_panel = self._panel(
            parent,
            key="ga",
            title=f"{self.scope_label}: GA: поиск кандидатных конфигураций",
            hint="чистый поиск без AHP-дублирования",
        )
        self.genetic_optimization_tab = GeneticOptimizationTab(
            ga_panel.content,
            run_genetic_optimization_use_case,
            run_genetic_ahp_ranking_use_case=run_genetic_ahp_ranking_use_case,
            energy_tab=energy_tab,
            data_root=data_root,
            profile_service=self.profile_service,
            initial_analysis_scope=self.scope,
            lock_analysis_scope=True,
            on_candidate_pool_ready=self._receive_candidate_pool_from_ga,
        )
        self.genetic_optimization_tab.pack(fill="both", expand=True)
        self._add_adaptive_panel(parent, ga_panel, weight=2)

        ahp_panel = self._panel(
            parent,
            key="ahp",
            title=f"{self.scope_label}: AHP-анализ конфигураций",
            hint="ранжирование альтернатив",
            initially_open=False,
        )
        self.configuration_selection_tab = ConfigurationSelectionTab(
            ahp_panel.content,
            self.crud,
            profile_service=self.profile_service,
            initial_analysis_scope=self.scope,
            lock_analysis_scope=True,
            on_result_ready=self._receive_ahp_result,
        )
        self.configuration_selection_tab.pack(fill="both", expand=True)
        self._add_adaptive_panel(parent, ahp_panel, weight=2)

        criteria_panel = self._panel(
            parent,
            key="criteria",
            title=f"{self.scope_label}: Pareto и критериальное обоснование",
            hint="компромиссы и доминирование",
            initially_open=False,
        )
        self.criteria_importance_tab = CriteriaImportanceTab(
            criteria_panel.content,
            self.crud,
            profile_service=self.profile_service,
            initial_analysis_scope=self.scope,
            lock_analysis_scope=True,
            on_result_ready=self._receive_pareto_result,
        )
        self.criteria_importance_tab.pack(fill="both", expand=True)
        self._add_adaptive_panel(parent, criteria_panel, weight=2)

        hybrid_panel = self._panel(
            parent,
            key="hybrid",
            title=f"{self.scope_label}: гибридная итоговая оценка",
            hint="сводка GA, AHP и расхождений",
            initially_open=False,
        )
        self.hybrid_decision_assessment_tab = HybridDecisionAssessmentTab(
            hybrid_panel.content,
            run_genetic_optimization_use_case,
            run_genetic_ahp_ranking_use_case=run_genetic_ahp_ranking_use_case,
            energy_tab=energy_tab,
            data_root=data_root,
            profile_service=self.profile_service,
            initial_analysis_scope=self.scope,
            lock_analysis_scope=True,
        )
        self.hybrid_decision_assessment_tab.pack(fill="both", expand=True)
        self._add_adaptive_panel(parent, hybrid_panel, weight=2)

    def _receive_candidate_pool_from_ga(
        self,
        analysis_scope: str,
        candidates,
        metadata: Mapping[str, Any],
    ) -> None:
        if analysis_scope != self.scope:
            return
        count = int(metadata.get("count") or len(candidates or []))
        source_label = f"top-{count} GA-кандидатов ({self.scope_label})"
        snapshot = self.candidate_pool_service.replace(
            self.scope,
            candidates or [],
            source_label=source_label,
            source_method="ga",
        )
        if hasattr(self, "configuration_selection_tab"):
            self.configuration_selection_tab.load_candidate_pool(
                snapshot.candidates,
                source_label=source_label,
            )
        self.decision_state["candidate_pool_snapshot"] = snapshot
        self.decision_state["ahp_result"] = None
        self.decision_state["pareto_result"] = None
        self.decision_state["hybrid_result"] = None
        if hasattr(self, "criteria_importance_tab"):
            self.criteria_importance_tab.load_candidate_pool(
                snapshot.candidates,
                source_label=source_label,
            )
        if hasattr(self, "hybrid_decision_assessment_tab"):
            self.hybrid_decision_assessment_tab.load_candidate_pool(
                snapshot.candidates,
                source_label=source_label,
            )
            self.hybrid_decision_assessment_tab.clear_method_results()

    def _receive_ahp_result(self, analysis_scope: str, report: Mapping[str, Any]) -> None:
        if analysis_scope != self.scope:
            return
        self.decision_state["ahp_result"] = report
        if hasattr(self, "hybrid_decision_assessment_tab"):
            self.hybrid_decision_assessment_tab.update_ahp_result(report)

    def _receive_pareto_result(self, analysis_scope: str, report: Mapping[str, Any]) -> None:
        if analysis_scope != self.scope:
            return
        self.decision_state["pareto_result"] = report
        if hasattr(self, "hybrid_decision_assessment_tab"):
            self.hybrid_decision_assessment_tab.update_pareto_result(report)

    def _add_adaptive_panel(
        self,
        parent: ttk.Panedwindow,
        panel: CollapsiblePanel,
        *,
        weight: int,
    ) -> None:
        """Add a panel to a Panedwindow and bind disclosure state to sash sizes.

        A collapsed block should not leave behind an empty manually-resized
        pane.  The panel remains in the Panedwindow, but its pane is reduced to
        header height and the remaining opened siblings receive the released
        vertical space.
        """

        parent.add(panel, weight=weight)
        group = self._get_adaptive_group(parent)
        group["panels"].append({"panel": panel, "weight": max(int(weight), 1)})
        panel.set_toggle_callback(lambda _panel, pane=parent: self._schedule_rebalance_pane(pane))

    def _get_adaptive_group(self, pane: ttk.Panedwindow) -> dict[str, Any]:
        for group in self._adaptive_pane_groups:
            if group["pane"] is pane:
                return group
        group = {"pane": pane, "panels": []}
        self._adaptive_pane_groups.append(group)
        return group

    def _schedule_rebalance_pane(self, pane: ttk.Panedwindow) -> None:
        self._schedule_adaptive_after_idle(lambda pane=pane: self._rebalance_adaptive_pane(pane))
        self._schedule_adaptive_after(80, lambda pane=pane: self._rebalance_adaptive_pane(pane))

    def _schedule_rebalance_all_adaptive_panes(self) -> None:
        self._schedule_adaptive_after_idle(self._rebalance_all_adaptive_panes)
        self._schedule_adaptive_after(80, self._rebalance_all_adaptive_panes)

    def _schedule_adaptive_after_idle(self, callback) -> None:
        try:
            self._adaptive_after_ids.append(self.after_idle(callback))
        except tk.TclError:
            return

    def _schedule_adaptive_after(self, delay_ms: int, callback) -> None:
        try:
            self._adaptive_after_ids.append(self.after(delay_ms, callback))
        except tk.TclError:
            return

    def _cancel_adaptive_rebalance_callbacks(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        for after_id in self._adaptive_after_ids:
            try:
                self.after_cancel(after_id)
            except tk.TclError:
                continue
        self._adaptive_after_ids.clear()

    def _rebalance_all_adaptive_panes(self) -> None:
        for group in self._adaptive_pane_groups:
            self._rebalance_adaptive_pane(group["pane"])

    def _rebalance_adaptive_pane(self, pane: ttk.Panedwindow) -> None:
        group = self._get_adaptive_group(pane)
        entries = group["panels"]
        if len(entries) <= 1:
            return

        try:
            pane.update_idletasks()
            total_height = pane.winfo_height()
            if total_height <= 80:
                return

            sash_gap = 6 * (len(entries) - 1)
            collapsed_heights: list[int | None] = []
            open_weight = 0
            collapsed_total = 0
            for entry in entries:
                panel = entry["panel"]
                if panel.opened:
                    collapsed_heights.append(None)
                    open_weight += entry["weight"]
                else:
                    height = self._collapsed_panel_height(panel)
                    collapsed_heights.append(height)
                    collapsed_total += height

            available_for_open = max(total_height - collapsed_total - sash_gap, 0)
            heights: list[int] = []
            used_open_height = 0
            opened_seen = 0
            opened_total = sum(1 for value in collapsed_heights if value is None)
            for index, value in enumerate(collapsed_heights):
                if value is not None:
                    heights.append(value)
                    continue

                opened_seen += 1
                entry_weight = entries[index]["weight"]
                if open_weight <= 0:
                    height = 0
                elif opened_seen == opened_total:
                    height = max(available_for_open - used_open_height, 0)
                else:
                    height = int(available_for_open * entry_weight / open_weight)
                    used_open_height += height
                heights.append(max(height, self._minimum_open_panel_height(total_height, opened_total)))

            self._apply_sash_positions(pane, heights)
        except tk.TclError:
            return

    def _collapsed_panel_height(self, panel: CollapsiblePanel) -> int:
        try:
            panel.header.update_idletasks()
            return max(panel.header.winfo_reqheight() + 8, 34)
        except tk.TclError:
            return 38

    def _minimum_open_panel_height(self, total_height: int, opened_total: int) -> int:
        if opened_total <= 0:
            return 0
        if total_height < 420:
            return 80
        return 120

    def _apply_sash_positions(self, pane: ttk.Panedwindow, heights: list[int]) -> None:
        position = 0
        for index, height in enumerate(heights[:-1]):
            position += max(int(height), 1)
            pane.sashpos(index, position)

    def _panel(
        self,
        parent: tk.Misc,
        *,
        key: str,
        title: str,
        hint: str,
        initially_open: bool = True,
    ) -> CollapsiblePanel:
        panel = CollapsiblePanel(parent, title=title, header_hint=hint, initially_open=initially_open)
        self.panels[key] = panel
        return panel

    def _build_cost_relation_table(self, parent: tk.Misc) -> None:
        text = tk.Label(
            parent,
            text=(
                "Матрица фиксирует не место строки в интерфейсе, а экономический смысл. "
                "Разовая бессрочная лицензия ближе к CAPEX, подписка — к OPEX; "
                "миграция и тестирование остаются разовыми операционными работами."
            ),
            wraplength=760,
            justify="left",
            background=parent.cget("background") if "background" in parent.keys() else WORKSPACE,
            foreground=TEXT,
        )
        text.pack(fill="x", padx=6, pady=(0, 6))

        tree = ttk.Treeview(
            parent,
            columns=("item", "scope", "expense", "reason"),
            show="headings",
            height=7,
        )
        headings = {
            "item": "Что учитывается",
            "scope": "ПО/ТО",
            "expense": "CAPEX/OPEX",
            "reason": "Почему так",
        }
        widths = {"item": 260, "scope": 80, "expense": 110, "reason": 520}
        for column in ("item", "scope", "expense", "reason"):
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor="w")
        for row in COST_RELATION_ROWS:
            tree.insert("", "end", values=row)
        tree.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    def _force_panel_backgrounds(self) -> None:
        """Re-apply backgrounds after all nested legacy widgets are created."""

        for panel in self.panels.values():
            force_panel_backgrounds(panel.content)

    def refresh_all(self) -> None:
        self.capex_tab.refresh_all()
        self.opex_tab.refresh_all()
        self._force_panel_backgrounds()

    def open_all(self) -> None:
        for panel in self.panels.values():
            panel.open()
        self._schedule_rebalance_all_adaptive_panes()

    def close_analysis(self) -> None:
        for key in ("ga", "ahp", "criteria", "hybrid"):
            panel = self.panels.get(key)
            if panel is not None:
                panel.close()
        self._schedule_rebalance_all_adaptive_panes()


class TechnicalSolutionAreaWorkspaceTab(SolutionAreaWorkspaceTab):
    def __init__(self, parent: tk.Misc, **kwargs: Any) -> None:
        super().__init__(parent, scope=ANALYSIS_SCOPE_TECHNICAL, **kwargs)


class SoftwareSolutionAreaWorkspaceTab(SolutionAreaWorkspaceTab):
    def __init__(self, parent: tk.Misc, **kwargs: Any) -> None:
        super().__init__(parent, scope=ANALYSIS_SCOPE_SOFTWARE, **kwargs)


__all__ = [
    "SolutionAreaWorkspaceTab",
    "TechnicalSolutionAreaWorkspaceTab",
    "SoftwareSolutionAreaWorkspaceTab",
]
