from __future__ import annotations


def require_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"Поле '{field_name}' не должно быть пустым")
    return cleaned


def parse_int(value: str, field_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Поле '{field_name}' должно быть целым числом") from exc


def parse_float(value: str, field_name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Поле '{field_name}' должно быть числом") from exc
