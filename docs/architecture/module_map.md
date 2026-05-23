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
        INFRA[ИТ-инфраструктура]
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
    INFRA --> UC
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

- вкладки `CAPEX`, `OPEX`, `Электроэнергия`, `NPV`, `Экспорт`, `AHP` и `Анализ важности критериев` — это видимая часть системы;
- `application` связывает пользовательские действия с расчётами и хранением данных; переходная нормализация `scope`/`component_type`, профили ПО/ТО, адаптация к `CandidateConfiguration` и сборка `DecisionReport` выполняются здесь, а не в UI;
- `domain` содержит математику и модели;
- `infrastructure` хранит состояние и формирует выходные файлы, включая JSON/Markdown/CSV-представления `DecisionReport`;
- `tools/catalog_parser` обслуживает внешний справочник оборудования и не входит в runtime GUI.
