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
- `architecture/roadmap_patch_sequence_status.md` — сверка первых трёх пунктов предложенной devctl-очередности с уже выполненными Этапами 1–3 перед переходом к Этапу 9;
- `architecture/conceptual_cohesion_completion_summary.md` — итоговая сводка выполнения roadmap, закрытых общих правил, контрольных вопросов и дальнейших направлений;
- `architecture/current_subject_schema.md` — переходная предметная схема текущих CAPEX/OPEX-категорий, будущих `scope`/`component_type` и связей вкладок с данными;
- `architecture/analysis_scope_profiles.md` — прикладные профили анализа ПО/ТО: категории, критерии, ограничения, веса и правила объяснения;
- `architecture/candidate_configurations.md` — единый формат кандидатных альтернатив для AHP, GA, GA + AHP, анализа важности критериев и будущего DecisionReport;
- `architecture/tco_npv_bridge.md` — TCO-модель и финансовый мост между CAPEX/OPEX/электроэнергией и NPV;
- `architecture/decision_report.md` — единый итоговый отчёт выбора: JSON/Markdown/CSV, объяснение победителя, риски и защитный сценарий;
- `architecture/legacy_infrastructure_tab.md` — роль старой вкладки ИТ-инфраструктуры как вспомогательной песочницы, namespace `legacy_infrastructure:` и границы участия в аналитике;
- `architecture/future_solution_component_editor_variant_c.md` — будущая идея варианта C: универсальный редактор `SolutionComponent` после завершения roadmap;
- `architecture/solution_component_editor_roadmap.md` — детальный roadmap-чеклист пошаговой реализации универсального редактора компонентов;
- `architecture/solution_component_contract.md` — документационный контракт `SolutionComponent`: поля, стоимость, метрики, допуски и связи с общей аналитической моделью;
- `architecture/solution_component_runtime_storage.md` — runtime-хранение `SolutionComponent`, demo-fixture, статусы черновиков и совместимость экспорта;
- `architecture/solution_component_editor_mvp.md` — MVP пользовательского редактора компонентов решения, профильная форма и предпросмотр нормализации;
- `architecture/solution_component_financial_integration.md` — финансовая интеграция компонентов редактора с TCO, энергией, NPV и предупреждениями C5;
- `architecture/solution_component_analytics_integration.md` — аналитическая интеграция компонентов редактора с CandidateConfiguration, AHP, GA и GA + AHP;
- `architecture/solution_component_decision_report_export.md` — интеграция компонентов редактора с DecisionReport, JSON/Markdown/CSV-экспортом и объяснением исключённых черновиков;
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
- `demo/ga_ahp_defense_guide.md` — сценарий показа генетического подбора и связки ГА + AHP, текст выступления и ответы на вопросы;
- `demo/demo_data_contract.md` — контракт демо-данных, учебных примеров и регрессионных инвариантов после этапа 8.

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
5. `architecture/roadmap_patch_sequence_status.md`
6. `architecture/conceptual_cohesion_completion_summary.md`
7. `architecture/current_subject_schema.md`
8. `architecture/analysis_scope_profiles.md`
9. `architecture/candidate_configurations.md`
10. `architecture/tco_npv_bridge.md`
11. `architecture/decision_report.md`
12. `architecture/legacy_infrastructure_tab.md`
13. `architecture/future_solution_component_editor_variant_c.md`
14. `architecture/solution_component_editor_roadmap.md`
15. `architecture/solution_component_contract.md`
16. `architecture/solution_component_runtime_storage.md`
17. `architecture/solution_component_editor_mvp.md`
18. `architecture/solution_component_financial_integration.md`
19. `architecture/solution_component_analytics_integration.md`
20. `architecture/solution_component_decision_report_export.md`
21. `architecture/system_schema.md`
22. `architecture/module_map.md`
23. `architecture/data_map.md`
24. `research/genetic_algorithm.md`
25. `research/ideas/genetic_algorithm_plain_manifest.md`
26. `research/ideas/future_independent_ga_ahp_assessment.md`
27. `research/ideas/future_hybrid_ga_ahp_score.md`
28. `demo/ga_ahp_defense_guide.md`
29. `demo/demo_data_contract.md`
30. `traceability.md`
