# Qt energy screen

Патч `qt-energy-screen` переносит раздел `Электроэнергия` в новый Qt-контур.

## Что сделано

- Добавлен экран `EnergyScreen`.
- Добавлен `EnergyPresenter` как граница UI и application services.
- Расчёт использует существующий `CalculateElectricityCostsUseCase`.
- Экран показывает ключевые вводы и итоговые карточки.
- Таблица оборудования скрыта в раскрываемом блоке.
- Параметры 24/7 скрыты в блоке `Параметры`.
- Последний результат доступен через `QtAppPresenter`.

## UX-правила

- На экране нет длинных видимых label.
- Основной action один: `Рассчитать`.
- Таблица не перегружает стартовый вид.
- Пустое состояние предлагает загрузить демо.
- Подробные объяснения находятся в tooltip.

## Связь с будущими экранами

`QtAppPresenter.get_electricity_cost()` и `QtAppPresenter.get_electricity_profile()` нужны будущим NPV и Export экранам. Это сохраняет совместимость с прежним Tk flow, где energy tab отдавал `get_electricity_cost()` и `get_electricity_profile()`.
