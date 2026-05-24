from __future__ import annotations

import logging
import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from application.services.analysis_scope_profile_service import AnalysisScopeProfileService
from application.services.cost_aggregation_service import CostAggregationService
from application.services.decision_report_service import DecisionReportService
from application.services.decision_demo_service import DecisionDemoDataService
from application.services.electricity_cost_service import ElectricityCostService
from application.services.equipment_service import EquipmentService
from application.services.genetic_optimization_service import GeneticOptimizationService
from application.services.genetic_ahp_ranking_service import GeneticAhpRankingService
from application.services.npv_report_service import NPVReportService
from application.services.solution_component_runtime_service import SolutionComponentRuntimeService
from application.use_cases import (
    BuildDecisionReportUseCase,
    BuildNpvReportUseCase,
    CalculateElectricityCostsUseCase,
    ExportCostReportUseCase,
    RunGeneticOptimizationUseCase,
    RunGeneticAhpRankingUseCase,
    LoadDemoDatasetUseCase,
    PrepareCostSummaryUseCase,
)
from infrastructure.exporters.decision_report_exporter import (
    export_decision_report_csv,
    export_decision_report_json,
    export_decision_report_markdown,
)
from infrastructure.logging import configure_logging
from infrastructure.repositories.json_entity_repository import JsonEntityRepository
from infrastructure.repositories.treeview_crud_repository import TreeviewCrudRepository
from infrastructure.storage import JsonFileStorage
from shared.constants import ANALYSIS_SCOPE_TECHNICAL
from ui.tabs import (
    capex_tab,
    configuration_selection_tab,
    criteria_importance_tab,
    energy_tab,
    export_tab,
    genetic_optimization_tab,
    infrastructure_tab,
    solution_component_editor_tab,
    npv_tab,
    opex_tab,
)

logger = logging.getLogger(__name__)


class CalculatorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.repo_root = Path(__file__).resolve().parents[1]
        self.data_root = self.repo_root / "data"
        self.log_path = configure_logging(repo_root=self.repo_root)
        self._is_shutting_down = False
        logger.info("Инициализация настольного приложения. Лог-файл: %s", self.log_path)

        self.title("Калькулятор стоимости ИТ-инфраструктуры")
        self.geometry("1280x720")
        self.protocol("WM_DELETE_WINDOW", self.shutdown)

        storage = JsonFileStorage()
        runtime_entities_path = self.data_root / "generated" / "runtime_entities.json"
        self.entity_repository = JsonEntityRepository(runtime_entities_path, storage)
        self.crud = TreeviewCrudRepository(self.entity_repository)
        self.equipment_service = EquipmentService(self.entity_repository)
        self.cost_aggregation_service = CostAggregationService()
        self.analysis_scope_profile_service = AnalysisScopeProfileService()
        self.decision_demo_data_service = DecisionDemoDataService(self.analysis_scope_profile_service)
        self.solution_component_runtime_service = SolutionComponentRuntimeService(self.entity_repository)

        self.prepare_cost_summary_use_case = PrepareCostSummaryUseCase(
            self.cost_aggregation_service,
            self.equipment_service,
            self.solution_component_runtime_service,
        )
        self.export_cost_report_use_case = ExportCostReportUseCase(
            self.prepare_cost_summary_use_case, self.cost_aggregation_service
        )
        self.load_demo_dataset_use_case = LoadDemoDatasetUseCase(storage, self.equipment_service)
        self.calculate_electricity_costs_use_case = CalculateElectricityCostsUseCase(
            ElectricityCostService()
        )
        self.build_npv_report_use_case = BuildNpvReportUseCase(NPVReportService())
        self.build_decision_report_use_case = BuildDecisionReportUseCase(
            DecisionReportService()
        )
        self.genetic_optimization_service = GeneticOptimizationService(
            self.entity_repository,
            profile_service=self.analysis_scope_profile_service,
        )
        self.run_genetic_optimization_use_case = RunGeneticOptimizationUseCase(
            self.genetic_optimization_service
        )
        self.run_genetic_ahp_ranking_use_case = RunGeneticAhpRankingUseCase(
            GeneticAhpRankingService(self.genetic_optimization_service)
        )

        self.demo_status_var = tk.StringVar(
            value="Демо-данные не загружаются автоматически. При необходимости используйте кнопку выше."
        )
        self._build_toolbar()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.technical_equipment_tab = capex_tab.TechnicalEquipmentTab(
            self.notebook, self.equipment_service
        )
        self.capex_tab = self.technical_equipment_tab
        self.capital_costs_tab = self.technical_equipment_tab
        self.notebook.add(self.technical_equipment_tab, text="ТО")

        self.software_tab = capex_tab.SoftwareTab(self.notebook, self.equipment_service)
        self.notebook.add(self.software_tab, text="ПО")

        self.opex_tab = opex_tab.OpexTab(self.notebook, self.equipment_service)
        self.operational_costs_tab = self.opex_tab
        self.notebook.add(self.opex_tab, text="Операционные затраты")

        self.solution_component_editor_tab = solution_component_editor_tab.SolutionComponentEditorTab(
            self.notebook, self.solution_component_runtime_service
        )
        self.notebook.add(self.solution_component_editor_tab, text="Редактор компонентов")

        self.it_infrastructure_tab = infrastructure_tab.ITInfrastructureTab(
            self.notebook, self.crud
        )
        self.notebook.add(self.it_infrastructure_tab, text="ИТ-песочница (архив)")

        self.energy_tab = energy_tab.EnergyTab(
            self.notebook,
            self.equipment_service,
            self.calculate_electricity_costs_use_case,
        )
        self.electricity_tab = self.energy_tab
        self.notebook.add(self.energy_tab, text="Электроэнергия")

        self.export_tab = export_tab.ExportTab(self.notebook, self)
        self.notebook.add(self.export_tab, text="Экспорт")

        self.npv_tab = npv_tab.NPVTab(
            self.notebook,
            self.build_npv_report_use_case,
            cost_summary_provider=self.update_total_costs,
        )
        self.notebook.add(self.npv_tab, text="NPV-анализ")

        self.genetic_optimization_tab = genetic_optimization_tab.GeneticOptimizationTab(
            self.notebook,
            self.run_genetic_optimization_use_case,
            run_genetic_ahp_ranking_use_case=self.run_genetic_ahp_ranking_use_case,
            energy_tab=self.energy_tab,
            data_root=self.data_root,
            profile_service=self.analysis_scope_profile_service,
        )
        self.notebook.add(self.genetic_optimization_tab, text="Генетический подбор")

        self.configuration_selection_tab = configuration_selection_tab.ConfigurationSelectionTab(
            self.notebook,
            self.crud,
            profile_service=self.analysis_scope_profile_service,
        )
        self.ahp_tab = self.configuration_selection_tab
        self.notebook.add(self.configuration_selection_tab, text="AHP-анализ конфигураций")

        self.criteria_importance_tab = criteria_importance_tab.CriteriaImportanceTab(
            self.notebook,
            self.crud,
            profile_service=self.analysis_scope_profile_service,
        )
        self.notebook.add(self.criteria_importance_tab, text="Обоснование выбора ИТ-решения")

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        self.update_total_costs()
        self.export_tab.update_summary()
        logger.info("Настольное приложение инициализировано")

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self, padding=(12, 10))
        toolbar.pack(fill="x")

        ttk.Button(toolbar, text="Загрузить демо-данные", command=self.load_demo_data).pack(
            side="left"
        )
        ttk.Label(toolbar, textvariable=self.demo_status_var).pack(side="left", padx=(12, 0))
        ttk.Button(toolbar, text="Выход", command=self.shutdown).pack(side="right")

    def _has_runtime_data(self) -> bool:
        return any(rows for rows in self.entity_repository.entities.values())

    def _refresh_runtime_views(self) -> None:
        self.technical_equipment_tab.refresh_all()
        self.software_tab.refresh_all()
        self.opex_tab.refresh_all()
        self.solution_component_editor_tab.refresh_components()
        self.energy_tab.update_equipment_table()
        self.update_total_costs()
        self.export_tab.update_summary()
        logger.debug("UI-представления обновлены после изменения runtime-данных")

    def load_demo_data(self) -> None:
        demo_dataset_path = self.data_root / "fixtures" / "demo_dataset.json"
        if not demo_dataset_path.exists():
            logger.error("Не найден файл демо-данных: %s", demo_dataset_path)
            messagebox.showerror(
                "Файл не найден", f"Не найден файл демо-данных:\n{demo_dataset_path}", parent=self
            )
            self.demo_status_var.set("Не удалось загрузить демо-данные: fixture-файл отсутствует.")
            return

        if self._has_runtime_data():
            should_replace = messagebox.askyesno(
                "Заменить текущие данные?",
                "В runtime-хранилище уже есть записи. Загрузить демонстрационный набор вместо них?",
                parent=self,
            )
            if not should_replace:
                logger.warning("Пользователь отменил замену runtime-данных демо-набором")
                self.demo_status_var.set("Загрузка демо-данных отменена пользователем.")
                return

        self.load_demo_dataset_use_case.execute(demo_dataset_path)
        self._refresh_runtime_views()

        demo_decision_payload = self.decision_demo_data_service.build(self.entity_repository.entities)
        scoped_payloads = demo_decision_payload.get("scoped_payloads", {})
        self.configuration_selection_tab.load_scoped_demo_payloads(
            scoped_payloads,
            default_scope=ANALYSIS_SCOPE_TECHNICAL,
            message=(
                "Демо-конфигурации собраны из того же набора данных, который был загружен "
                "во вкладки ТО и ПО. Переключатель области позволяет отдельно анализировать "
                "техническое и программное обеспечение."
            ),
        )
        self.criteria_importance_tab.load_case_data(
            demo_decision_payload["criteria_case"],
            scoped_cases={
                scope: payload.get("criteria_case", {})
                for scope, payload in scoped_payloads.items()
            },
            analysis_scope=ANALYSIS_SCOPE_TECHNICAL,
            message=(
                "Демонстрационный кейс обоснования выбора ИТ-решения сформирован на основе текущего набора ТО. "
                "Переключите область на ПО, чтобы применить Pareto/критериальный анализ к лицензиям."
            ),
        )

        self.demo_status_var.set(
            "Демо-данные загружены вручную из data/fixtures/demo_dataset.json."
        )
        logger.info("Демо-данные загружены через GUI: %s", demo_dataset_path)
        messagebox.showinfo(
            "Готово",
            "Демонстрационный набор данных загружен, а связанные вкладки обновлены аутентичными сценариями.",
            parent=self,
        )

    def on_tab_change(self, _event) -> None:
        current_index = self.notebook.index("current")
        if current_index == self.notebook.index(self.solution_component_editor_tab):
            self.solution_component_editor_tab.refresh_components()
        if current_index == self.notebook.index(self.energy_tab):
            logger.debug("Активирована вкладка электроэнергии, запускается обновление")
            self.energy_tab.update_equipment_table()
            self.update_total_costs()
        self.export_tab.update_summary()

    def update_total_costs(self) -> dict:
        totals = self.prepare_cost_summary_use_case.execute(
            electricity_cost=self.energy_tab.get_electricity_cost(),
            electricity_profile=self.energy_tab.get_electricity_profile(),
        )
        logger.debug("Итоги затрат пересчитаны: %s", totals)
        return totals

    def export_to_csv(self) -> dict:
        export_path = self.data_root / "generated" / "total_costs.csv"
        logger.info("Запрос на экспорт CSV: %s", export_path)
        totals = self.export_cost_report_use_case.execute(
            export_path,
            electricity_cost=self.energy_tab.get_electricity_cost(),
            electricity_profile=self.energy_tab.get_electricity_profile(),
        )
        self.export_tab.update_summary()
        logger.info("CSV-экспорт завершён: %s", export_path)
        return totals

    def export_decision_report(self) -> dict[str, Path]:
        export_root = self.data_root / "generated"
        report_path = export_root / "decision_report.json"
        markdown_path = export_root / "decision_report.md"
        csv_path = export_root / "decision_report_candidates.csv"

        last_genetic_result = getattr(self.genetic_optimization_tab, "last_result", None)
        genetic_result = None
        genetic_ahp_result = None
        if isinstance(last_genetic_result, dict):
            if "ahp_report" in last_genetic_result and "genetic_optimization" in last_genetic_result:
                genetic_ahp_result = last_genetic_result
            else:
                genetic_result = last_genetic_result

        report = self.build_decision_report_use_case.execute(
            project={
                "id": "current-it-solution-choice",
                "title": "Итоговый отчёт выбора ИТ-решения",
                "goal": (
                    "Объяснить выбор ИТ-решения через компоненты, стоимость владения, "
                    "альтернативы, AHP, GA, GA + AHP и NPV."
                ),
            },
            entities=self.entity_repository.entities,
            cost_totals=self.update_total_costs(),
            criteria_importance_result=getattr(self.criteria_importance_tab, "analysis_report", None),
            genetic_result=genetic_result,
            genetic_ahp_result=genetic_ahp_result,
            npv_report=getattr(self.npv_tab, "last_report", None),
            metadata={"source": "desktop_export_tab"},
        )

        json_path = export_decision_report_json(report, report_path)
        md_path = export_decision_report_markdown(report, markdown_path)
        table_path = export_decision_report_csv(report, csv_path)
        logger.info(
            "DecisionReport экспортирован: json=%s markdown=%s csv=%s",
            json_path,
            md_path,
            table_path,
        )
        return {"json": json_path, "markdown": md_path, "csv": table_path}

    def shutdown(self) -> None:
        if self._is_shutting_down:
            return

        self._is_shutting_down = True
        logger.info("Запрошено завершение настольного приложения")

        try:
            if hasattr(self, "npv_tab") and hasattr(self.npv_tab, "shutdown"):
                self.npv_tab.shutdown()

            try:
                self.quit()
            except tk.TclError:
                logger.debug("Пропущен вызов quit(): Tcl уже завершён")

            try:
                self.destroy()
            except tk.TclError:
                logger.debug("Пропущен вызов destroy(): окно уже уничтожено")
        finally:
            logging.shutdown()
            os._exit(0)
