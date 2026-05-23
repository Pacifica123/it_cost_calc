# Архитектурная схема

Ниже показана укрупнённая схема проекта после завершения roadmap концептуальной связанности. Она отражает не только технические слои, но и основной поток данных от ввода компонентов до итогового отчёта выбора.

```mermaid
flowchart LR
    U[Пользователь] --> UI[UI на Tkinter]

    subgraph UIX[Вкладки]
        CAPEX[CAPEX / ТО / ПО]
        OPEX[OPEX]
        ENERGY[Электроэнергия]
        NPV[NPV]
        ANALYSIS[AHP / GA / анализ важности]
        EXPORT[Экспорт]
        SANDBOX[ИТ-песочница]
    end

    UI --> UIX
    CAPEX --> APP[Прикладной слой]
    OPEX --> APP
    ENERGY --> APP
    NPV --> APP
    ANALYSIS --> APP
    EXPORT --> APP
    SANDBOX --> APP

    APP --> NORM[RuntimeEntityNormalizationService]
    NORM --> PROFILE[AnalysisScopeProfileService]
    PROFILE --> CAND[CandidateConfigurationService]
    CAND --> TCO[TCOModelService]
    TCO --> NPVS[NpvReportService]
    CAND --> METHODS[AHP / GA / GA + AHP / критерии]
    METHODS --> DRS[DecisionReportService]
    NPVS --> DRS
    TCO --> DRS

    APP --> DOM[Предметная логика]
    DOM --> AHPD[AHP]
    DOM --> CID[Анализ важности]
    DOM --> GA[Оптимизация]
    DOM --> FIN[Финансы]

    APP --> INF[Инфраструктурные адаптеры]
    INF --> RUNTIME[(Runtime JSON)]
    INF --> REPORTS[(CSV / JSON / Markdown отчёты)]
    INF --> LOGS[(Логи)]

    FIX[(Демо и регрессионные фикстуры)] --> APP
    FIX --> DCTRL[DemoControlScenarioService]
    DCTRL --> DRS

    PARSER[Инструмент обновления каталога] --> CATALOG[(Каталог оборудования)]
    CATALOG --> APP
```

## Смысл схемы

- `UI` принимает действия пользователя и показывает результаты, но не является владельцем предметной модели.
- `RuntimeEntityNormalizationService` делает старые и новые записи сопоставимыми через `scope` и `component_type`.
- `AnalysisScopeProfileService` хранит различия ПО и ТО: категории, критерии, ограничения, веса и объяснения.
- `CandidateConfigurationService` превращает компоненты и результаты методов в общий пул альтернатив.
- `TCOModelService` связывает CAPEX, OPEX, внедрение, сопровождение и электроэнергию.
- `NpvReportService` использует рассчитанные затраты как основу финансовой оценки, а не как ручной повторный ввод.
- `DecisionReportService` собирает компоненты, стоимость, альтернативы, результаты методов, NPV, риски и предупреждения в единый итоговый артефакт.
- `ИТ-песочница` сохраняет legacy-сценарий свободных статей, но её записи отделены от строгой аналитики.

## Главный архитектурный принцип

Основное приложение работает с подготовленными компонентами, профилями, альтернативами и отчётами. Сетевой сбор данных, HTML-разбор и исследовательские эксперименты не смешиваются с пользовательским runtime приложения.
