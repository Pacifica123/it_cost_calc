from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ui_qt.text_rules import assert_short_text

FieldKind = Literal["text", "int", "float"]
Parser = Callable[[str, str], Any]


def parse_text(value: str, label: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{label}: пусто")
    return value


def parse_optional_text(value: str, _label: str) -> str:
    return value.strip()


def parse_int(value: str, label: str) -> int:
    value = value.strip().replace(" ", "")
    if not value:
        raise ValueError(f"{label}: пусто")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{label}: нужно число") from exc


def parse_float(value: str, label: str) -> float:
    value = value.strip().replace(" ", "").replace(",", ".")
    if not value:
        raise ValueError(f"{label}: пусто")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{label}: нужно число") from exc


_KIND_PARSERS: dict[FieldKind, Parser] = {
    "text": parse_text,
    "int": parse_int,
    "float": parse_float,
}


@dataclass(frozen=True)
class FieldSpec:
    """Compact schema for a Qt record form field.

    Visible labels and placeholders are checked against the UI density rule.
    Longer explanations belong to ``help`` and become tooltips in the dialog.
    """

    name: str
    label: str
    kind: FieldKind = "text"
    default: Any = ""
    placeholder: str = ""
    help: str = ""
    required: bool = True
    parser: Parser | None = None

    def __post_init__(self) -> None:
        assert_short_text(self.label, field=f"FieldSpec[{self.name}].label")
        if self.placeholder:
            assert_short_text(
                self.placeholder,
                field=f"FieldSpec[{self.name}].placeholder",
            )
        if self.kind not in _KIND_PARSERS:
            raise ValueError(f"Unknown field kind: {self.kind!r}")

    def parse(self, value: str) -> Any:
        if not self.required and not value.strip():
            return ""
        parser = self.parser or _KIND_PARSERS[self.kind]
        return parser(value, self.label)

    def as_dialog_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "placeholder": self.placeholder,
            "help": self.help,
            "required": self.required,
            "parser": self.parser,
        }


def coerce_field_spec(raw: FieldSpec | Mapping[str, Any]) -> FieldSpec:
    if isinstance(raw, FieldSpec):
        return raw
    return FieldSpec(
        name=str(raw["name"]),
        label=str(raw["label"]),
        kind=raw.get("kind", "text"),
        default=raw.get("default", ""),
        placeholder=str(raw.get("placeholder", "")),
        help=str(raw.get("help", "")),
        required=bool(raw.get("required", True)),
        parser=raw.get("parser"),
    )


def normalize_field_specs(fields: Sequence[FieldSpec | Mapping[str, Any]]) -> list[FieldSpec]:
    return [coerce_field_spec(field) for field in fields]


def payload_from_strings(
    fields: Sequence[FieldSpec | Mapping[str, Any]],
    raw_values: Mapping[str, str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in normalize_field_specs(fields):
        payload[field.name] = field.parse(str(raw_values.get(field.name, "")))
    return payload


DEFAULT_ENTITY_FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("name", "Название", placeholder="Название"),
    FieldSpec("quantity", "Кол.", kind="int", default=1, placeholder="1"),
    FieldSpec("price", "Цена", kind="float", default=0, placeholder="0"),
    FieldSpec("monthly_cost", "Ежемес.", kind="float", default=0, placeholder="0"),
    FieldSpec("one_time_cost", "Разово", kind="float", default=0, placeholder="0"),
)


def default_entity_fields() -> list[FieldSpec]:
    return list(DEFAULT_ENTITY_FIELD_SPECS)
