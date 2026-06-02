"""Composite ПО/ТО workspace tabs with collapsible and resizable panels."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Mapping

from application.services.analysis_scope_profile_service import AnalysisScopeProfileService
from application.services.equipment_service import EquipmentService
from application.use_cases.run_genetic_ahp_ranking import RunGeneticAhpRankingUseCase
from application.use_cases.run_genetic_optimization import RunGeneticOptimizationUseCase
from shared.constants import ANALYSIS_SCOPE_SOFTWARE, ANALYSIS_SCOPE_TECHNICAL
from ui.tabs.capex_tab import CapexTab, SoftwareTab, TechnicalEquipmentTab
from ui.tabs.configuration_selection_tab import ConfigurationSelectionTab
from ui.tabs.criteria_importance_tab import CriteriaImportanceTab
from ui.tabs.genetic_optimization_tab import GeneticOptimizationTab
from ui.tabs.opex_tab import OpexTab
from ui.theme import WORKSPACE, force_panel_backgrounds
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
        self.panels: dict[str, CollapsiblePanel] = {}

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
        header = ttk.Frame(self, padding=(12, 10), style="WorkspaceHeader.TFrame")
        header.pack(fill="x")
        header.columnconfigure(1, weight=1)
        ttk.Label(
            header,
            text=self.scope_title,
            font=("TkDefaultFont", 12, "bold"),
            style="WorkspaceHeader.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text=self._workspace_hint(),
            wraplength=720,
            justify="left",
            style="WorkspaceHeader.TLabel",
        ).grid(row=0, column=1, padx=(16, 10), sticky="ew")
        ttk.Button(header, text="Свернуть анализ", command=self.close_analysis).grid(
            row=0, column=2, padx=(0, 6), sticky="e"
        )
        ttk.Button(header, text="Раскрыть всё", command=self.open_all).grid(
            row=0, column=3, sticky="e"
        )

        main = ttk.Panedwindow(self, orient="horizontal", style="Workspace.TPanedwindow")
        main.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        data_side = ttk.Panedwindow(main, orient="vertical", style="Workspace.TPanedwindow")
        analysis_side = ttk.Panedwindow(main, orient="vertical", style="Workspace.TPanedwindow")
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
                analysis_side.sashpos(0, int(height * 0.84))
                analysis_side.sashpos(1, int(height * 0.92))
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
        parent.add(capex_panel, weight=1)

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
        parent.add(opex_panel, weight=1)

        relation_panel = self._panel(
            parent,
            key="cost_relation",
            title="Матрица ПО/ТО × CAPEX/OPEX",
            hint="финансовая классификация",
            initially_open=False,
        )
        self._build_cost_relation_table(relation_panel.content)
        parent.add(relation_panel, weight=1)

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
            title=f"{self.scope_label}: генетический подбор и GA + AHP",
            hint="область зафиксирована вкладкой",
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
        )
        self.genetic_optimization_tab.pack(fill="both", expand=True)
        parent.add(ga_panel, weight=2)

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
        )
        self.configuration_selection_tab.pack(fill="both", expand=True)
        parent.add(ahp_panel, weight=2)

        criteria_panel = self._panel(
            parent,
            key="criteria",
            title=f"{self.scope_label}: обоснование выбора ИТ-решений",
            hint="Pareto и критерии",
            initially_open=False,
        )
        self.criteria_importance_tab = CriteriaImportanceTab(
            criteria_panel.content,
            self.crud,
            profile_service=self.profile_service,
            initial_analysis_scope=self.scope,
            lock_analysis_scope=True,
        )
        self.criteria_importance_tab.pack(fill="both", expand=True)
        parent.add(criteria_panel, weight=2)

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
        text = ttk.Label(
            parent,
            text=(
                "Матрица фиксирует не место строки в интерфейсе, а экономический смысл. "
                "Разовая бессрочная лицензия ближе к CAPEX, подписка — к OPEX; "
                "миграция и тестирование остаются разовыми операционными работами."
            ),
            wraplength=760,
            justify="left",
            style="PanelBody.TLabel",
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

    def close_analysis(self) -> None:
        for key in ("ga", "ahp", "criteria"):
            panel = self.panels.get(key)
            if panel is not None:
                panel.close()


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
