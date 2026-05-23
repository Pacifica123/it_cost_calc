# Карта данных

Ниже показано, какие группы данных используются в проекте и как они движутся между каталогами.

```mermaid
flowchart LR
    FIX[data/fixtures/demo_dataset.json] --> RUNTIME[data/generated/runtime_entities.json]
    RUNTIME --> RNORM[application: normalization scope/component_type]
    RNORM --> REPORTS[data/generated/*.csv]
    RUNTIME --> LOGS[data/generated/logs/]

    RAW[data/examples/parser/*.json] --> CNORM[data/examples/catalog/normalized_dns_catalog.json]
    CNORM --> CATALOG[data/generated/catalog/equipment_catalog.json]
    CATALOG --> APP[Основное приложение]

    EXAHP[data/examples/ahp/*.json] --> TESTS[tests/integration + tests/regression]
    FIX --> TESTS
```

## Группы данных

### 1. Демонстрационные и эталонные данные
- `data/fixtures/demo_dataset.json` — демонстрационный набор для ручной загрузки;
- `data/examples/ahp/` — эталонные наборы для AHP;
- `data/examples/optimization/` — примеры для оптимизационных сценариев;
- `data/examples/parser/` — сохранённые снимки парсинга.

### 2. Рабочие runtime-данные
- `data/generated/runtime_entities.json` — текущее состояние пользовательских сущностей;
- `data/generated/logs/` — логи приложения;
- `data/generated/catalog/` — рабочий каталог оборудования;
- `data/generated/ahp/` — выходные файлы расчётов и отчётов.

### 3. Каталог оборудования
- `data/examples/catalog/normalized_dns_catalog.json` — пример нормализованного каталога;
- `data/generated/catalog/equipment_catalog.json` — рабочий каталог, формируемый инструментом обновления.

## Таблица артефактов

| Артефакт | Где лежит | Кто создаёт | Кто использует |
|---|---|---|---|
| Демонстрационный набор | `data/fixtures/demo_dataset.json` | разработчик/исходная фикстура | GUI, тесты, CLI загрузки |
| Runtime-сущности | `data/generated/runtime_entities.json` | приложение | GUI, сервисы, экспорт |
| CSV-отчёт | `data/generated/` или пользовательский путь | сценарий экспорта | пользователь |
| Каталог оборудования | `data/generated/catalog/equipment_catalog.json` | `tools/catalog_parser` | основное приложение, тесты |
| Пример каталога | `data/examples/catalog/normalized_dns_catalog.json` | подготовленный пример | тесты, документация |
| Логи | `data/generated/logs/it_cost_calc.log` | приложение | разработчик |


## Runtime-нормализация после этапа 2

`data/generated/runtime_entities.json` и старые фикстуры могут не содержать явных полей `scope` и `component_type`. Прикладной слой достраивает эти признаки через `RuntimeEntityNormalizationService` перед тем, как записи попадут в сервисы стоимости, GA/AHP-сценарии или экспорт.

Переходная логика не переносится во вкладки интерфейса: UI передаёт категорию и пользовательские поля, а application-слой решает, относится ли запись к `technical`, `software`, `implementation` или `common`, и какой `component_type` можно безопасно назначить.

Особое правило действует для `client`: если у записи нет `client_seats` и явного `component_type=workstation`, она не считается рабочим местом автоматически. Поэтому периферия остаётся частью клиентского контура, но не закрывает ограничения по пользователям.

## Предметный контракт категорий

Физическое движение данных описывает этот файл, а смысл текущих CAPEX/OPEX-категорий закреплён в `current_subject_schema.md`. Там указано, какие категории относятся к техническому обеспечению, программному обеспечению, внедрению и общему контуру, а также как они должны перейти к будущим признакам `scope` и `component_type`.

Короткая связь между картами:

| Уровень | Где описан | Что фиксирует |
|---|---|---|
| Физическое хранение | `data_map.md` | какие JSON/CSV/лог-файлы создаются и кто их использует |
| Предметный смысл | `current_subject_schema.md` | что означают категории `server`, `client`, `licenses`, `migration`, `backup` и другие |
| Будущая эволюция | `conceptual_cohesion_roadmap.md` | как перейти от категорий к `scope`, `component_type`, профилям анализа и общей модели альтернатив |

## Главный принцип

В репозитории разделены:
- **фикстуры и примеры** — то, на чём воспроизводятся сценарии;
- **runtime-данные** — то, что создаётся в ходе работы приложения;
- **каталог оборудования** — внешний справочник, обновляемый отдельным инструментом.
