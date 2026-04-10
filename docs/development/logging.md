# Логирование

В проекте используется единая настройка логирования через модуль `src/it_cost_calc/infrastructure/logging.py`.

## Что настроено

- логгеры создаются по имени модуля (`logging.getLogger(__name__)`);
- консольный вывод показывает сообщения от выбранного уровня и выше;
- файловый лог всегда сохраняет сообщения уровня `DEBUG` и выше;
- логи сохраняются в `data/generated/logs/it_cost_calc.log`;
- повторный вызов `configure_logging()` не дублирует обработчики.

## Как использовать

```python
from it_cost_calc.infrastructure.logging import configure_logging

configure_logging()
```

Для конкретного модуля:

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Модуль запущен")
```

## Переменная окружения

Для консольного уровня логирования можно использовать переменную окружения:

```bash
IT_COST_LOG_LEVEL=DEBUG python scripts/run_app.py
```

Допустимые уровни: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
