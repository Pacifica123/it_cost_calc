# Связь между разделами работы и кодовой базой

Этот файл нужен как мост между содержанием дипломной работы и инженерной структурой репозитория.

| Раздел работы / тема | Что реализует в проекте | Где лежит код | Где взять пример входных данных |
|---|---|---|---|
| Расчёт капитальных затрат | Ввод состава оборудования и подсчёт CAPEX | `src/it_cost_calc/ui/tabs/capex_tab.py`, `src/it_cost_calc/application/services/equipment_service.py`, `src/it_cost_calc/application/services/cost_aggregation_service.py` | `data/fixtures/demo_dataset.json` |
| Расчёт операционных затрат | Учёт разовых и периодических OPEX | `src/it_cost_calc/ui/tabs/opex_tab.py`, `src/it_cost_calc/application/services/equipment_service.py` | `data/fixtures/demo_dataset.json` |
| Оценка затрат на электроэнергию | Профили энергопотребления и расчёт стоимости | `src/it_cost_calc/ui/tabs/energy_tab.py`, `src/it_cost_calc/application/services/electricity_cost_service.py` | `data/fixtures/demo_dataset.json` |
| NPV-анализ | Расчёт инвестиционной эффективности | `src/it_cost_calc/ui/tabs/npv_tab.py`, `src/it_cost_calc/application/services/npv_report_service.py`, `src/it_cost_calc/domain/finance/` | данные вводятся вручную; сценарий можно воспроизвести через `scripts/run_app.py` |
| AHP-анализ конфигураций | Сравнение альтернатив по критериям и ограничениям | `src/it_cost_calc/ui/tabs/configuration_selection_tab.py`, `src/it_cost_calc/domain/decision/ahp/` | `data/examples/ahp/test_confs.json`, `data/examples/ahp/test_criteria.json` |
| Анализ важности критериев | Обоснование выбора решения по относительной важности критериев | `src/it_cost_calc/ui/tabs/criteria_importance_tab.py`, `src/it_cost_calc/domain/decision/criteria_importance/` | встроенный кейс в модуле + регрессионный пример `tests/regression/test_criteria_importance_budgeting_case.py` |
| Экспорт результатов | Выгрузка сводных расчётов в CSV | `src/it_cost_calc/ui/tabs/export_tab.py`, `src/it_cost_calc/application/use_cases/export_cost_report.py`, `src/it_cost_calc/infrastructure/exporters/csv_exporter.py` | `data/fixtures/demo_dataset.json` |
| Каталог оборудования и его актуализация | Нормализация данных о реально существующем оборудовании | `tools/catalog_parser/`, `scripts/update_equipment_catalog.py` | `data/examples/parser/`, `data/examples/catalog/normalized_dns_catalog.json` |
| Оптимизационные эвристики | Экспериментальный подбор конфигураций | `src/it_cost_calc/domain/optimization/` | `data/examples/optimization/` |

## Как использовать этот файл

- для пояснения на защите, какой раздел работы опирается на какой код;
- для навигации по репозиторию без долгого чтения всех каталогов;
- для связи между текстом работы, демонстрационными данными и инженерной реализацией.
