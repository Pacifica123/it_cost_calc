# Карта модулей

Эта карта показывает не только каталоги репозитория, но и смысловые связи между ними.
Подробное описание каждого модуля вынесено в `modules.md`.

```mermaid
flowchart TB
    subgraph UI[Пользовательский интерфейс]
        CAPEX[CAPEX]
        OPEX[OPEX]
        ENERGY[Электроэнергия]
        NPV[NPV]
        INFRA[ИТ-песочница]
        SCUI[Редактор компонентов]
        AHPUI[AHP-анализ]
        CIUI[Анализ важности критериев]
        EXPORTUI[Экспорт]
        DRUI[DecisionReport export]
    end

    subgraph APP[Прикладной слой]
        ES[EquipmentService]
        NORM[RuntimeEntityNormalizationService]
        PROFILE[AnalysisScopeProfileService]
        CAND[CandidateConfigurationService]
        CAS[CostAggregationService]
        ECS[ElectricityCostService]
        NPVS[NpvReportService]
        DRS[DecisionReportService]
        DCTRL[DemoControlScenarioService]
        SCNORM[SolutionComponentNormalizationService]
        SCRUNTIME[SolutionComponentRuntimeService]
        SCFIN[SolutionComponentFinancialIntegrationService]
        SCAN[SolutionComponentAnalyticsIntegrationService]
        UC[Use cases]
    end

    subgraph DOM[Предметная логика]
        MODELS[Модели]
        FIN[Финансы]
        AHPD[AHP]
        CID[Важность критериев]
        OPT[Оптимизация]
    end

    subgraph INF[Инфраструктура]
        REPO[Runtime-репозитории]
        STORE[JSON storage]
        CSV[CSV exporter]
        DREXP[DecisionReport exporters]
        LOG[Logging]
    end

    subgraph TOOLS[Вспомогательные инструменты]
        PARSER[Catalog parser]
        CATALOG[Каталог оборудования]
    end

    CAPEX --> ES
    OPEX --> ES
    ENERGY --> ECS
    NPV --> NPVS
    EXPORTUI --> UC
    DRUI --> UC
    INFRA --> REPO
    SCUI --> SCRUNTIME
    AHPUI --> UC
    CIUI --> UC

    ES --> NORM
    CAS --> NORM
    NORM --> PROFILE
    PROFILE --> CAND
    CAND --> AHPD
    CAND --> CID
    CAND --> OPT
    CAND --> MODELS
    CAND --> DRS
    DCTRL --> NORM
    DCTRL --> CAND
    DCTRL --> DRS
    SCNORM --> PROFILE
    SCRUNTIME --> SCNORM
    SCFIN --> SCRUNTIME
    SCFIN --> NPVS
    SCAN --> SCNORM
    SCAN --> CAND
    SCAN --> OPT
    SCAN --> AHPD
    SCNORM --> DRS
    SCRUNTIME --> REPO
    NORM --> MODELS
    ES --> REPO
    CAS --> REPO
    ECS --> ES
    NPVS --> FIN
    NPVS --> DRS
    UC --> AHPD
    UC --> CID
    UC --> CAS

    REPO --> STORE
    UC --> CSV
    UC --> DREXP
    UC --> LOG

    PARSER --> CATALOG
    CATALOG --> UC
    MODELS --> ES
    MODELS --> AHPD
    MODELS --> CID
    OPT --> UC
```

## Практическое чтение карты

- вкладки `CAPEX`/`ТО`/`ПО`, `OPEX`, `Электроэнергия`, `NPV`, `Экспорт`, `AHP` и `Анализ важности критериев` — это видимая часть основного аналитического сценария;
- `ИТ-песочница` остаётся вспомогательной legacy-вкладкой: она пишет свободные статьи в runtime-хранилище с префиксом `legacy_infrastructure:`, но не является источником строгой предметной модели;
- `Редактор компонентов` работает с `SolutionComponent` для нестандартных и составных случаев, а не заменяет быстрые вкладки ПО/ТО;
- `application` связывает пользовательские действия с расчётами и хранением данных; переходная нормализация `scope`/`component_type`, профили ПО/ТО, адаптация к `CandidateConfiguration`, demo/control-сценарий и сборка `DecisionReport` выполняются здесь, а не в UI;
- `domain` содержит математику и модели;
- `infrastructure` хранит состояние и формирует выходные файлы, включая JSON/Markdown/CSV-представления `DecisionReport`;
- `tools/catalog_parser` обслуживает внешний справочник оборудования и не входит в runtime GUI.

## Где находятся ключевые связующие сервисы

| Сервис | Слой | Роль в общей цепочке |
|---|---|---|
| `RuntimeEntityNormalizationService` | `application` | Достраивает `scope`, `component_type`, `client_seats` и legacy-признаки. |
| `AnalysisScopeProfileService` | `application` | Хранит профили ПО/ТО: критерии, ограничения, категории и подсказки. |
| `CandidateConfigurationService` | `application` | Приводит runtime-записи и результаты методов к единому формату альтернатив. |
| `SolutionComponentNormalizationService` | `application` | Нормализует компоненты редактора, рассчитывает warnings и признаки допуска. |
| `SolutionComponentRuntimeService` | `application` | Хранит `SolutionComponent` в `entities.solution_components` и отделяет strict-компоненты от draft. |
| `SolutionComponentFinancialIntegrationService` | `application` | Передаёт расходы компонентов редактора в TCO и NPV-базу без искусственного эффекта. |
| `SolutionComponentAnalyticsIntegrationService` | `application` | Передаёт profile-ready компоненты в AHP/GA через `CandidateConfiguration`. |
| `TCOModelService` | `application` | Собирает стоимость владения и финансовую базу NPV. |
| `DecisionReportService` | `application` | Формирует единый итоговый отчёт выбора. |
| `DemoControlScenarioService` | `application` | Проверяет, что демонстрационный набор проходит всю цепочку без отдельной смысловой модели. |
| `decision_report_exporter` | `infrastructure` | Записывает отчёт в JSON, Markdown и CSV-срез альтернатив. |

Эти сервисы являются главной причиной, почему предметная логика больше не размазана по вкладкам интерфейса.
