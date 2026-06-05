# Связь между разделами работы и кодовой базой

Этот файл нужен как мост между содержанием дипломной работы и инженерной структурой репозитория.

| Раздел работы / тема | Что реализует в проекте | Где лежит код | Где взять пример входных данных |
|---|---|---|---|
| Расчёт капитальных затрат | Ввод состава оборудования и подсчёт CAPEX | `src/ui/tabs/capex_tab.py`, `src/application/services/equipment_service.py`, `src/application/services/cost_aggregation_service.py` | `data/fixtures/demo_dataset.json` |
| Расчёт операционных затрат | Учёт разовых и периодических OPEX | `src/ui/tabs/opex_tab.py`, `src/application/services/equipment_service.py` | `data/fixtures/demo_dataset.json` |
| Оценка затрат на электроэнергию | Профили энергопотребления и расчёт стоимости | `src/ui/tabs/energy_tab.py`, `src/application/services/electricity_cost_service.py` | `data/fixtures/demo_dataset.json` |
| NPV-анализ | Расчёт инвестиционной эффективности | `src/ui/tabs/npv_tab.py`, `src/application/services/npv_report_service.py`, `src/domain/finance/` | данные вводятся вручную; сценарий можно воспроизвести через `scripts/run_app.py` |
| AHP-анализ конфигураций | Сравнение альтернатив по критериям и ограничениям | `src/ui/tabs/configuration_selection_tab.py`, `src/domain/decision/ahp/` | `data/examples/ahp/test_confs.json`, `data/examples/ahp/test_criteria.json` |
| Анализ важности критериев | Обоснование выбора решения по относительной важности критериев | `src/ui/tabs/criteria_importance_tab.py`, `src/domain/decision/criteria_importance/` | встроенный кейс в модуле + регрессионный пример `tests/regression/test_criteria_importance_budgeting_case.py` |
| Генетический подбор конфигурации | Автоматический поиск допустимого состава оборудования по ограничениям и весам критериев | `src/ui/tabs/genetic_optimization_tab.py`, `src/application/services/genetic_optimization_service.py`, `src/application/use_cases/run_genetic_optimization.py`, `src/domain/optimization/ga.py` | `data/fixtures/demo_dataset.json`, `scripts/run_ga_demo.py`, `src/domain/optimization/pipeline_knapsack.py` |
| Общий пул альтернатив ПО/ТО | Единый пул `CandidateConfiguration`, который формирует GA или demo-загрузка и который затем используют AHP, Pareto, Hybrid и DecisionReport | `src/application/services/scoped_candidate_pool_service.py`, `src/ui/tabs/solution_area_workspace_tab.py`, `src/application/services/candidate_configuration_service.py` | `data/fixtures/demo_dataset.json`, кнопка передачи top-N из GA в общий пул |
| Legacy-сервис ГА + AHP | Совместимый машинный слой старого сценария, не основной UI-путь после рефакторинга | `src/application/services/genetic_ahp_ranking_service.py`, `src/application/use_cases/run_genetic_ahp_ranking.py` | тест `tests/unit/test_genetic_ahp_ranking_service.py` |
| Гибридная итоговая оценка | Сводная витрина согласованности GA, AHP и Pareto без повторного запуска методов | `src/application/services/hybrid_decision_assessment_service.py`, `src/ui/tabs/hybrid_decision_assessment_tab.py`, `src/ui/tabs/solution_area_workspace_tab.py` | общий пул текущей области, последний AHP-результат, последний Pareto-результат |
| Экспорт результатов | Выгрузка сводных расчётов, CSV-кандидатов и DecisionReport с источником общего пула | `src/ui/tabs/export_tab.py`, `src/application/use_cases/export_cost_report.py`, `src/infrastructure/exporters/csv_exporter.py`, `src/infrastructure/exporters/decision_report_exporter.py` | `data/fixtures/demo_dataset.json` |
| Demo/control-сценарий | Проверка, что демонстрационный набор проходит `scope/component_type`, `CandidateConfiguration`, TCO и `DecisionReport` | `src/application/services/demo_control_scenario_service.py`, `scripts/run_demo_control_scenario.py` | `data/fixtures/demo_dataset.json`, `data/fixtures/regression/demo_control_invariants.json` |
| Итоговый отчёт выбора | Сведение компонентов, стоимости, альтернатив, аналитики, NPV, рисков и предупреждений в единый DecisionReport | `src/application/services/decision_report_service.py`, `src/application/use_cases/build_decision_report.py`, `src/infrastructure/exporters/decision_report_exporter.py`, `src/ui/tabs/export_tab.py` | `data/fixtures/demo_dataset.json`, `scripts/run_demo_control_scenario.py --check-only` |
| Каталог оборудования и его актуализация | Нормализация данных о реально существующем оборудовании | `tools/catalog_parser/`, `scripts/update_equipment_catalog.py` | `data/examples/parser/`, `data/examples/catalog/normalized_dns_catalog.json` |
| Оптимизационные эвристики | Универсальное ядро поиска подмножества item-ов, нормализованная fitness-функция, top-solutions и демонстрационный knapsack-сценарий | `src/domain/optimization/` | `data/examples/optimization/`, `scripts/run_ga_demo.py`, `tests/regression/test_ga_demo_pipeline_regression.py` |

## Как использовать этот файл

- для пояснения на защите, какой раздел работы опирается на какой код;
- для навигации по репозиторию без долгого чтения всех каталогов;
- для связи между текстом работы, демонстрационными данными и инженерной реализацией.

## Связь с итоговой архитектурной цепочкой

Для дипломного описания удобно показывать не отдельные вкладки, а сквозной поток:

```text
CAPEX/OPEX/энергия → scope/component_type → CandidateConfiguration → TCO → GA → общий пул → AHP/Pareto → Hybrid → NPV → DecisionReport
```

Таблица выше помогает связать каждый участок этого потока с кодом и воспроизводимыми входными данными.
