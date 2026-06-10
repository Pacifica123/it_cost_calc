# Калькулятор стоимости ИТ-инфраструктуры

`IT Cost Calculator` — настольное приложение на `PySide6/Qt` для расчёта затрат на ИТ-инфраструктуру, анализа инвестиционной эффективности и сравнения альтернативных конфигураций оборудования.

Проект объединяет несколько связанных задач:
- расчёт **CAPEX**;
- расчёт **OPEX**;
- оценку затрат на **электроэнергию**;
- **NPV-анализ** инвестиционного сценария;
- **AHP-анализ конфигураций**;
- **анализ важности критериев** для обоснования выбора решения;
- **генетический подбор конфигурации** по ограничениям и критериям;
- связку **ГА + AHP** для ранжирования найденных кандидатных конфигураций;
- единый **DecisionReport** для объяснения итогового выбора, рисков и ограничений;
- вспомогательную **ИТ-песочницу** для свободных черновых статей без смешивания со строгой аналитической моделью;
- подготовку и обновление **каталога оборудования** через отдельный вспомогательный инструмент.

## Основные возможности

1. Ввести состав инфраструктуры и получить сводную стоимость.
2. Рассчитать энергопотребление оборудования с учётом режима работы.
3. Построить NPV-отчёт по инвестиционному сценарию.
4. Сравнить альтернативные конфигурации методом анализа иерархий.
5. Обосновать выбор решения на основе относительной важности критериев.
6. Запустить генетический подбор конфигурации по бюджету, мощности, категориям и весам критериев.
7. Ранжировать найденные ГА конфигурации через AHP.
8. Выгрузить итоговые расчёты в CSV.
9. Сформировать DecisionReport в JSON, Markdown и CSV-срезе альтернатив.
10. При необходимости загрузить демонстрационный набор данных вручную.
11. Обновить каталог оборудования через отдельный CLI-инструмент.

## Быстрый запуск

### Требования к Python

Для запуска из исходников нужен Python 3.11+. Если установлен Python 3.10 или ниже, используйте готовую сборку `exe`/`AppImage` либо обновите интерпретатор. Проверка окружения:

```bash
python scripts/doctor.py
```

### Установка зависимостей
```bash
pip install -r requirements/base.txt
```

### Запуск приложения
```bash
python scripts/run_app.py
```

### Загрузка демонстрационного набора данных
```bash
python scripts/load_demo_data.py
```

### Обновление каталога оборудования из example-снимков
```bash
python scripts/update_equipment_catalog.py --mode examples
```

### Запуск демонстрационного ГА-сценария
```bash
python scripts/run_ga_demo.py
```

### Проверка сквозного demo/control-сценария
```bash
python -B scripts/run_demo_control_scenario.py --check-only
```

Эта команда не использует `pytest`, не пишет cache-файлы и проверяет, что `data/fixtures/demo_dataset.json` проходит цепочку `scope/component_type → CandidateConfiguration → TCO → DecisionReport`.

### Запуск тестов
```bash
pytest
```

### Release-сборки без Python у пользователя

```bash
pip install -r requirements/build.txt
```

Windows exe:

```powershell
python scripts/build_windows_exe.py
```

Linux AppImage:

```bash
bash scripts/build_linux_appimage.sh
```

Подробности: `docs/development/release_builds.md`.

## Структура репозитория

```text
.
├── README.md
├── pyproject.toml
├── requirements/
├── docs/
├── data/
├── scripts/
├── src/
├── tests/
└── tools/
```

## Что где находится

- `src/` — основной код приложения;
- `docs/` — инженерная, продуктовая, исследовательская и дипломная документация;
- `data/examples/` — воспроизводимые примеры данных;
- `data/fixtures/` — демонстрационные фикстуры, реестр ролей и регрессионные инварианты demo/control-сценария;
- `data/generated/` — генерируемые артефакты: каталоги, логи, отчёты;
- `tools/catalog_parser/` — отдельный инструмент обновления каталога оборудования.
- `src/ui_qt/` — основной Qt-интерфейс приложения;
- `src/it_cost_calc/` — только слой совместимости старого namespace; основная реализация не дублируется и хранится в `src/application`, `src/domain`, `src/infrastructure`, `src/shared`, `src/ui_qt`.

## Куда смотреть в документации

- обзор документации: `docs/README.md`
- архитектурная схема: `docs/architecture/system_schema.md`
- карта модулей: `docs/architecture/module_map.md`
- карта данных: `docs/architecture/data_map.md`
- сверка первых трёх пунктов devctl-очередности roadmap: `docs/architecture/roadmap_patch_sequence_status.md`
- итоговая сводка roadmap концептуальной связанности: `docs/architecture/conceptual_cohesion_completion_summary.md`
- текущая предметная схема CAPEX/OPEX и будущих признаков `scope`/`component_type`: `docs/architecture/current_subject_schema.md`
- профили анализа ПО/ТО, критерии и ограничения прикладного слоя: `docs/architecture/analysis_scope_profiles.md`
- единый формат кандидатных альтернатив: `docs/architecture/candidate_configurations.md`
- TCO-модель и связь CAPEX/OPEX/электроэнергии с NPV: `docs/architecture/tco_npv_bridge.md`
- единый итоговый отчёт выбора DecisionReport: `docs/architecture/decision_report.md`
- будущая идея варианта C как универсального редактора компонентов: `docs/architecture/future_solution_component_editor_variant_c.md`
- подробное описание модулей: `docs/architecture/modules.md`
- продуктовая идея: `docs/product/vision.md`
- генетический алгоритм: `docs/research/genetic_algorithm.md`
- сценарий демонстрации ГА + AHP: `docs/demo/ga_ahp_defense_guide.md`
- контракт демо-данных: `docs/demo/demo_data_contract.md`
- трассировка разделов работы и модулей кода: `docs/traceability.md`
- скриншоты интерфейса: `docs/screenshots.md`

## Разработка

Для локальной разработки:

```bash
pip install -r requirements/dev.txt
ruff check src tests scripts
black --check src tests scripts
pytest
```

Подробности вынесены в:
- `docs/development/setup.md`
- `docs/development/dev_tools.md`
- `docs/development/testing.md`
- `docs/development/logging.md`

## Логирование

В проекте используется централизованное логирование через `src/infrastructure/logging.py`.

- основной лог-файл: `data/generated/logs/it_cost_calc.log`;
- уровень консольного вывода задаётся переменной окружения `IT_COST_LOG_LEVEL`;
- отладочные логи не пишутся в корень репозитория.

## Текущая архитектурная логика после roadmap

После завершения roadmap проект следует читать как связанную систему поддержки выбора ИТ-решения:

```text
runtime-записи → scope/component_type → AnalysisScopeProfile → CandidateConfiguration → TCO/NPV → DecisionReport
```

Это означает, что вкладки CAPEX, OPEX, электроэнергии, NPV, AHP, GA и экспорта больше не описываются как независимые калькуляторы. Они остаются отдельными пользовательскими экранами, но работают вокруг общей предметной цепочки: компоненты нормализуются, превращаются в альтернативы, получают стоимость владения, анализируются разными методами и попадают в итоговый отчёт выбора.
