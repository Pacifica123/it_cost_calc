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
    end

    subgraph APP[Прикладной слой]
        ES[EquipmentService]
        CAS[CostAggregationService]
        ECS[ElectricityCostService]
        NPVS[NpvReportService]
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
    INFRA --> UC
    AHPUI --> UC
    CIUI --> UC

    ES --> REPO
    CAS --> REPO
    ECS --> ES
    NPVS --> FIN
    UC --> AHPD
    UC --> CID
    UC --> CAS

    REPO --> STORE
    UC --> CSV
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
- `application` связывает пользовательские действия с расчётами и хранением данных;
- `domain` содержит математику и модели;
- `infrastructure` хранит состояние и формирует выходные файлы;
- `tools/catalog_parser` обслуживает внешний справочник оборудования и не входит в runtime GUI.
