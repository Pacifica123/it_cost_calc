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
3. `architecture/system_schema.md`
4. `architecture/module_map.md`
5. `architecture/data_map.md`
6. `research/genetic_algorithm.md`
7. `research/ideas/genetic_algorithm_plain_manifest.md`
8. `research/ideas/future_independent_ga_ahp_assessment.md`
9. `research/ideas/future_hybrid_ga_ahp_score.md`
10. `demo/ga_ahp_defense_guide.md`
11. `traceability.md`
