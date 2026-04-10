from __future__ import annotations


def to_json_serializable(obj):
    import collections.abc

    # Позволяет расширять обработку конкретных классов
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {to_json_serializable(k): to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_serializable(el) for el in obj]
    # Обработка конкретного класса FuzzyValue
    if type(obj).__name__ == "FuzzyValue":
        return {"low": getattr(obj, "low", None), "high": getattr(obj, "high", None)}
    # Для других объектов пытаемся взять их __dict__ или поля dataclass
    if hasattr(obj, "__dict__"):
        return {
            k: to_json_serializable(v) for k, v in obj.__dict__.items() if not k.startswith("_")
        }
    if hasattr(obj, "__slots__"):  # если используются слоты
        return {slot: to_json_serializable(getattr(obj, slot)) for slot in obj.__slots__}
    # Для читаемых коллекций (NamedTuple и пр.)
    if isinstance(obj, collections.abc.Iterable):
        try:
            return [to_json_serializable(i) for i in obj]
        except Exception:
            pass
    # По умолчанию возвращаем строковое представление
    return str(obj)
