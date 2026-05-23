# Документация проекта

Этот каталог собран так, чтобы по нему можно было быстро понять:
- что делает проект;
- как устроен код;
- где лежат данные;
- как связаны инженерные модули и дипломные разделы;
- где находятся исходные академические материалы.

## Состав документации

### Архитектура
- `architecture/overview.md` — общий обзор решения;
- `architecture/conceptual_cohesion_manifest.md` — манифест концептуальной связанности данных, сущностей и аналитических методов;
- `architecture/conceptual_cohesion_roadmap.md` — дорожная карта дальнейших работ по связности модели, профилям ПО/ТО, TCO, аналитике и итоговому отчёту;
- `architecture/current_subject_schema.md` — переходная предметная схема текущих CAPEX/OPEX-категорий, будущих `scope`/`component_type` и связей вкладок с данными;
- `architecture/analysis_scope_profiles.md` — прикладные профили анализа ПО/ТО: категории, критерии, ограничения, веса и правила объяснения;
- `architecture/candidate_configurations.md` — единый формат кандидатных альтернатив для AHP, GA, GA + AHP, анализа важности критериев и будущего DecisionReport;
- `architecture/tco_npv_bridge.md` — TCO-модель и финансовый мост между CAPEX/OPEX/электроэнергией и NPV;
- `architecture/decision_report.md` — единый итоговый отчёт выбора: JSON/Markdown/CSV, объяснение победителя, риски и защитный сценарий;
- `architecture/system_schema.md` — укрупнённая архитектурная схема;
- `architecture/module_map.md` — карта модулей и связей между ними;
- `architecture/data_map.md` — карта данных и артефактов;
- `architecture/modules.md` — подробное описание инженерных модулей;
- `architecture/layers.md` — описание слоёв приложения;
- `architecture/repository_layout.md` — логика структуры репозитория;
- `architecture/catalog_boundary.md` — граница между приложением, каталогом и инструментом парсинга.

### Продукт
- `product/vision.md` — продуктовая идея и траектория развития.

### Разработка
- `development/setup.md` — запуск проекта и локальная работа;
- `development/dev_tools.md` — инструменты разработки;
- `development/testing.md` — тестовая стратегия;
- `development/logging.md` — логирование.

### Демонстрация и защита
- `demo/README.md` — обзор демонстрационных сценариев;
- `demo/ga_ahp_defense_guide.md` — сценарий показа генетического подбора и связки ГА + AHP, текст выступления и ответы на вопросы.

### Исследовательские материалы
- `research/README.md` — обзор исследовательской зоны;
- `research/ahp.md` — заметки по AHP;
- `research/criteria_importance.md` — анализ важности критериев;
- `research/genetic_algorithm.md` — генетический алгоритм подбора конфигурации и связка ГА + AHP;
- `research/parsing.md` — парсинг каталога оборудования;
- `research/ideas/genetic_algorithm_plain_manifest.md` — простое md-объяснение роли ГА;
- `research/ideas/future_independent_ga_ahp_assessment.md` — идея независимого сопоставления оценок ГА и AHP;
- `research/ideas/future_hybrid_ga_ahp_score.md` — идея гибридной итоговой оценки ГА и AHP.

### Материалы по работе
- `traceability.md` — связь между разделами работы, кодом и примерами данных;
- `screenshots.md` — скриншоты приложения;
- `thesis/README.md` — описание академических исходников;
- `thesis/raw/` — сами исходные материалы.

## Рекомендуемый маршрут чтения

Если нужно быстро войти в проект, удобно читать так:
1. `../README.md`
2. `architecture/overview.md`
3. `architecture/conceptual_cohesion_manifest.md`
4. `architecture/conceptual_cohesion_roadmap.md`
5. `architecture/current_subject_schema.md`
6. `architecture/analysis_scope_profiles.md`
7. `architecture/candidate_configurations.md`
8. `architecture/tco_npv_bridge.md`
9. `architecture/decision_report.md`
10. `architecture/system_schema.md`
11. `architecture/module_map.md`
12. `architecture/data_map.md`
13. `research/genetic_algorithm.md`
14. `research/ideas/genetic_algorithm_plain_manifest.md`
15. `research/ideas/future_independent_ga_ahp_assessment.md`
16. `research/ideas/future_hybrid_ga_ahp_score.md`
17. `demo/ga_ahp_defense_guide.md`
18. `traceability.md`
