from __future__ import annotations

import json
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
    export_solution_component_csv,
)
from infrastructure.logging import configure_logging
from infrastructure.repositories.json_entity_repository import JsonEntityRepository
from infrastructure.repositories.treeview_crud_repository import TreeviewCrudRepository
from infrastructure.storage import JsonFileStorage
from shared.constants import ANALYSIS_SCOPE_SOFTWARE, ANALYSIS_SCOPE_TECHNICAL
from ui.tabs import (
    energy_tab,
    export_tab,
    solution_area_workspace_tab,
    solution_component_editor_tab,
    npv_tab,
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

        # Energy is created before the ПО/ТО workspaces because the GA panel uses
        # it as a provider of equipment power lookup. The tab itself is added
        # later to preserve the target top-level order from the reorganization plan.
        self.energy_tab = energy_tab.EnergyTab(
            self.notebook,
            self.equipment_service,
            self.calculate_electricity_costs_use_case,
        )
        self.electricity_tab = self.energy_tab

        self.software_workspace_tab = solution_area_workspace_tab.SoftwareSolutionAreaWorkspaceTab(
            self.notebook,
            equipment_service=self.equipment_service,
            crud=self.crud,
            run_genetic_optimization_use_case=self.run_genetic_optimization_use_case,
            run_genetic_ahp_ranking_use_case=self.run_genetic_ahp_ranking_use_case,
            energy_tab=self.energy_tab,
            data_root=self.data_root,
            profile_service=self.analysis_scope_profile_service,
        )
        self.notebook.add(self.software_workspace_tab, text="Программное обеспечение")

        self.technical_workspace_tab = solution_area_workspace_tab.TechnicalSolutionAreaWorkspaceTab(
            self.notebook,
            equipment_service=self.equipment_service,
            crud=self.crud,
            run_genetic_optimization_use_case=self.run_genetic_optimization_use_case,
            run_genetic_ahp_ranking_use_case=self.run_genetic_ahp_ranking_use_case,
            energy_tab=self.energy_tab,
            data_root=self.data_root,
            profile_service=self.analysis_scope_profile_service,
        )
        self.notebook.add(self.technical_workspace_tab, text="Техническое обеспечение")

        # Compatibility aliases for services/tests/scripts that still expect the
        # old tab attributes while the top-level notebook is already reorganized.
        self.software_tab = self.software_workspace_tab.capex_tab
        self.technical_equipment_tab = self.technical_workspace_tab.capex_tab
        self.capex_tab = self.technical_equipment_tab
        self.capital_costs_tab = self.technical_equipment_tab
        self.software_opex_tab = self.software_workspace_tab.opex_tab
        self.technical_opex_tab = self.technical_workspace_tab.opex_tab
        self.opex_tab = self.software_opex_tab
        self.operational_costs_tab = self.opex_tab
        self.genetic_optimization_tab = self.technical_workspace_tab.genetic_optimization_tab
        self.configuration_selection_tab = self.technical_workspace_tab.configuration_selection_tab
        self.ahp_tab = self.configuration_selection_tab
        self.criteria_importance_tab = self.technical_workspace_tab.criteria_importance_tab
        self.workspace_tabs_by_scope = {
            ANALYSIS_SCOPE_SOFTWARE: self.software_workspace_tab,
            ANALYSIS_SCOPE_TECHNICAL: self.technical_workspace_tab,
        }
        self.genetic_optimization_tabs_by_scope = {
            scope: workspace.genetic_optimization_tab
            for scope, workspace in self.workspace_tabs_by_scope.items()
        }
        self.configuration_selection_tabs_by_scope = {
            scope: workspace.configuration_selection_tab
            for scope, workspace in self.workspace_tabs_by_scope.items()
        }
        self.criteria_importance_tabs_by_scope = {
            scope: workspace.criteria_importance_tab
            for scope, workspace in self.workspace_tabs_by_scope.items()
        }

        self.solution_component_editor_tab = solution_component_editor_tab.SolutionComponentEditorTab(
            self.notebook, self.solution_component_runtime_service
        )
        self.notebook.add(self.solution_component_editor_tab, text="Редактор компонентов")

        self.notebook.add(self.energy_tab, text="Электроэнергия")

        self.npv_tab = npv_tab.NPVTab(
            self.notebook,
            self.build_npv_report_use_case,
            cost_summary_provider=self.update_total_costs,
        )
        self.notebook.add(self.npv_tab, text="NPV-анализ")

        self.export_tab = export_tab.ExportTab(self.notebook, self)
        self.notebook.add(self.export_tab, text="Экспорт")

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


    def _refresh_solution_component_views(self) -> None:
        self.solution_component_editor_tab.refresh_components()
        self.update_total_costs()
        self.export_tab.update_summary()
        logger.debug("UI-представления обновлены после ручной конвертации sandbox-записи")

    def _has_runtime_data(self) -> bool:
        return any(rows for rows in self.entity_repository.entities.values())

    def _refresh_runtime_views(self) -> None:
        for workspace in self.workspace_tabs_by_scope.values():
            workspace.refresh_all()
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
        scoped_criteria_cases = {
            scope: payload.get("criteria_case", {})
            for scope, payload in scoped_payloads.items()
        }

        for scope, tab in self.configuration_selection_tabs_by_scope.items():
            scope_label = self.analysis_scope_profile_service.labels().get(scope, scope)
            tab.load_scoped_demo_payloads(
                scoped_payloads,
                default_scope=scope,
                message=(
                    f"Демо-конфигурации загружены внутрь рабочей области {scope_label}. "
                    "Расчёт AHP будет выполняться только по зафиксированной области этой вкладки."
                ),
            )

        for scope, tab in self.criteria_importance_tabs_by_scope.items():
            scope_label = self.analysis_scope_profile_service.labels().get(scope, scope)
            case_data = scoped_criteria_cases.get(scope) or demo_decision_payload["criteria_case"]
            tab.load_case_data(
                case_data,
                scoped_cases=scoped_criteria_cases,
                analysis_scope=scope,
                message=(
                    f"Демонстрационный Pareto/критериальный кейс загружен для области {scope_label}. "
                    "Экспорт будет видеть этот результат как часть комплексного отчёта после расчёта."
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
        components_csv_path = export_root / "decision_report_solution_components.csv"

        genetic_result, genetic_ahp_result, criteria_importance_result = (
            self._scoped_report_inputs()
        )

        report = self.build_decision_report_use_case.execute(
            project={
                "id": "current-it-solution-choice",
                "title": "Итоговый отчёт выбора ИТ-решения",
                "goal": (
                    "Объяснить выбор ИТ-решения через компоненты, стоимость владения, "
                    "альтернативы ПО/ТО, AHP, GA, GA + AHP и NPV."
                ),
            },
            entities=self.entity_repository.entities,
            cost_totals=self.update_total_costs(),
            criteria_importance_result=criteria_importance_result,
            genetic_result=genetic_result,
            genetic_ahp_result=genetic_ahp_result,
            npv_report=getattr(self.npv_tab, "last_report", None),
            metadata={
                "source": "desktop_export_tab",
                "scoped_analysis_state": self._scoped_analysis_state(),
            },
        )

        json_path = export_decision_report_json(report, report_path)
        md_path = export_decision_report_markdown(report, markdown_path)
        table_path = export_decision_report_csv(report, csv_path)
        components_table_path = export_solution_component_csv(report, components_csv_path)
        logger.info(
            "DecisionReport экспортирован: json=%s markdown=%s csv=%s components_csv=%s",
            json_path,
            md_path,
            table_path,
            components_table_path,
        )
        return {
            "json": json_path,
            "markdown": md_path,
            "csv": table_path,
            "components_csv": components_table_path,
        }

    def _scoped_report_inputs(self):
        genetic_results = {}
        genetic_ahp_results = {}
        criteria_results = {}

        for scope, workspace in self.workspace_tabs_by_scope.items():
            last_genetic_result = getattr(workspace.genetic_optimization_tab, "last_result", None)
            if isinstance(last_genetic_result, dict):
                if "ahp_report" in last_genetic_result and "genetic_optimization" in last_genetic_result:
                    genetic_ahp_results[scope] = last_genetic_result
                else:
                    genetic_results[scope] = last_genetic_result

            criteria_report = getattr(workspace.criteria_importance_tab, "analysis_report", None)
            if isinstance(criteria_report, dict):
                criteria_results[scope] = criteria_report

        return (
            {"by_scope": genetic_results} if genetic_results else None,
            {"by_scope": genetic_ahp_results} if genetic_ahp_results else None,
            {"by_scope": criteria_results} if criteria_results else None,
        )

    def _scoped_analysis_state(self) -> dict:
        state = {}
        for scope, workspace in self.workspace_tabs_by_scope.items():
            label = self.analysis_scope_profile_service.labels().get(scope, scope)
            ga_result = getattr(workspace.genetic_optimization_tab, "last_result", None)
            ahp_tab = workspace.configuration_selection_tab
            criteria_tab = workspace.criteria_importance_tab
            state[scope] = {
                "label": label,
                "ga_status": self._analysis_status(ga_result),
                "ahp_configurations": len(getattr(ahp_tab, "configurations", {}) or {}),
                "criteria_status": self._analysis_status(
                    getattr(criteria_tab, "analysis_report", None)
                ),
            }
        return state

    def _analysis_status(self, payload) -> str:
        if not payload:
            return "не рассчитано"
        if isinstance(payload, dict):
            status = payload.get("status")
            if status:
                return str(status)
            if "ahp_report" in payload and "genetic_optimization" in payload:
                return "GA + AHP рассчитано"
            return "рассчитано"
        return "есть данные"

    def build_export_dashboard_summary(self) -> str:
        totals = self.update_total_costs()
        lines = [
            "Живая сводка экспорта",
            "====================",
            f"CAPEX: {self._format_money(totals.get('total_capital', 0.0))}",
            f"OPEX разовый: {self._format_money(totals.get('total_operational_one_time', 0.0))}",
            f"OPEX периодический: {self._format_money(totals.get('total_operational_monthly', 0.0))}",
            f"Электроэнергия: {self._format_money(totals.get('electricity_costs', 0.0))}",
        ]
        tco = totals.get("tco")
        if isinstance(tco, dict):
            lines.extend(
                [
                    f"TCO за период: {self._format_money(tco.get('total_ownership_cost', 0.0))}",
                    f"Годовой OPEX: {self._format_money(tco.get('annual_opex', 0.0))}",
                ]
            )

        lines.append("")
        lines.append("Состояние аналитики по областям:")
        for scope, item in self._scoped_analysis_state().items():
            lines.append(
                f"- {item['label']}: GA={item['ga_status']}; "
                f"AHP-конфигураций={item['ahp_configurations']}; "
                f"обоснование={item['criteria_status']}."
            )

        npv_report = getattr(self.npv_tab, "last_report", None)
        lines.append("")
        if isinstance(npv_report, dict):
            lines.append(f"NPV: рассчитан, значение={self._format_money(npv_report.get('npv', 0.0))}.")
        else:
            lines.append("NPV: ещё не рассчитан; полный финансовый вывод будет неполным.")

        lines.append("")
        lines.append(
            "Частичный экспорт снабжается оговорками: отдельный CAPEX/OPEX/анализ "
            "не доказывает оптимальность всей конфигурации без остальных разделов."
        )
        return "\n".join(lines)

    def export_selected_fragment(self, mode: str) -> dict[str, Path]:
        export_root = self.data_root / "generated"
        export_root.mkdir(parents=True, exist_ok=True)

        if mode == "full_decision_report":
            return self.export_decision_report()
        if mode == "raw_costs_csv":
            self.export_to_csv()
            return {"csv": export_root / "total_costs.csv"}

        if mode == "cost_summary":
            return {"markdown": self._write_markdown_export(
                export_root / "cost_summary_context.md",
                "Сводка CAPEX/OPEX/TCO",
                self._cost_summary_markdown(),
            )}
        if mode == "technical_analysis":
            return {"markdown": self._write_markdown_export(
                export_root / "technical_analysis_context.md",
                "Частичный экспорт аналитики ТО",
                self._scope_analysis_markdown(ANALYSIS_SCOPE_TECHNICAL),
            )}
        if mode == "software_analysis":
            return {"markdown": self._write_markdown_export(
                export_root / "software_analysis_context.md",
                "Частичный экспорт аналитики ПО",
                self._scope_analysis_markdown(ANALYSIS_SCOPE_SOFTWARE),
            )}
        if mode == "npv":
            return {"markdown": self._write_markdown_export(
                export_root / "npv_context.md",
                "Частичный экспорт NPV",
                self._npv_markdown(),
            )}

        raise ValueError(f"Неизвестный режим экспорта: {mode}")

    def _write_markdown_export(self, path: Path, title: str, body: str) -> Path:
        text = f"# {title}\n\n{body.rstrip()}\n"
        path.write_text(text, encoding="utf-8")
        return path

    def _cost_summary_markdown(self) -> str:
        totals = self.update_total_costs()
        payload = json.dumps(totals, ensure_ascii=False, indent=2, default=str)
        return (
            "Этот фрагмент фиксирует только финансовую сводку. Он не подтверждает, "
            "что выбранная конфигурация лучшая по GA/AHP/Pareto, и не заменяет NPV.\n\n"
            "## Текущие итоги\n\n"
            f"- CAPEX: {self._format_money(totals.get('total_capital', 0.0))}\n"
            f"- OPEX разовый: {self._format_money(totals.get('total_operational_one_time', 0.0))}\n"
            f"- OPEX периодический: {self._format_money(totals.get('total_operational_monthly', 0.0))}\n"
            f"- Электроэнергия: {self._format_money(totals.get('electricity_costs', 0.0))}\n\n"
            "## Машиночитаемый снимок\n\n"
            "```json\n"
            f"{payload}\n"
            "```"
        )

    def _scope_analysis_markdown(self, scope: str) -> str:
        workspace = self.workspace_tabs_by_scope[scope]
        label = self.analysis_scope_profile_service.labels().get(scope, scope)
        ga_result = getattr(workspace.genetic_optimization_tab, "last_result", None)
        criteria_report = getattr(workspace.criteria_importance_tab, "analysis_report", None)
        ahp_count = len(getattr(workspace.configuration_selection_tab, "configurations", {}) or {})
        payload = {
            "scope": scope,
            "ga_or_ga_ahp": ga_result,
            "criteria_importance": criteria_report,
            "ahp_configurations_count": ahp_count,
        }
        return (
            f"Экспортируется только аналитика области {label}. "
            "Без соседней области, OPEX/TCO и NPV это не является полным обоснованием выбора.\n\n"
            f"- GA/GA+AHP: {self._analysis_status(ga_result)}\n"
            f"- AHP-конфигураций загружено: {ahp_count}\n"
            f"- Pareto/критериальное обоснование: {self._analysis_status(criteria_report)}\n\n"
            "## Машиночитаемый снимок\n\n"
            "```json\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}\n"
            "```"
        )

    def _npv_markdown(self) -> str:
        npv_report = getattr(self.npv_tab, "last_report", None)
        if not isinstance(npv_report, dict):
            return (
                "NPV ещё не рассчитан. Этот файл создан как контекстная пометка: "
                "для полного финансового вывода сначала нужно выполнить NPV-анализ."
            )
        return (
            "Экспортируется только NPV. Он показывает финансовую целесообразность сценария, "
            "но не объясняет, почему именно эта конфигурация лучшая по GA/AHP/Pareto.\n\n"
            "```json\n"
            f"{json.dumps(npv_report, ensure_ascii=False, indent=2, default=str)}\n"
            "```"
        )

    def _format_money(self, value) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        return f"{number:,.2f}".replace(",", " ")

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
