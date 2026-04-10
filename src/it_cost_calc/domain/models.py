from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, TypeVar


class ExpenseType(str, Enum):
    CAPITAL = "capital"
    OPERATIONAL = "operational"
    ELECTRICITY = "electricity"


class EquipmentCategory(str, Enum):
    SERVER = "server"
    CLIENT = "client"
    NETWORK = "network"
    LICENSES = "licenses"


class RelationOperator(str, Enum):
    EQUAL = "="
    GREATER = ">"


@dataclass(slots=True)
class CapitalItem:
    name: str
    quantity: int
    price: float
    category: EquipmentCategory | None = None
    expense_type: ExpenseType = ExpenseType.CAPITAL

    @property
    def total_cost(self) -> float:
        return float(self.quantity) * float(self.price)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        category: str | EquipmentCategory | None = None,
    ) -> "CapitalItem":
        resolved_category = None
        if category is not None:
            resolved_category = as_equipment_category(category)
        elif payload.get("category"):
            resolved_category = as_equipment_category(payload["category"])
        return cls(
            name=str(payload.get("name", "")),
            quantity=int(payload.get("quantity", 0)),
            price=float(payload.get("price", 0.0)),
            category=resolved_category,
        )


@dataclass(slots=True)
class OperationalExpense:
    name: str
    monthly_cost: float = 0.0
    one_time_cost: float = 0.0
    expense_type: ExpenseType = ExpenseType.OPERATIONAL

    @property
    def is_monthly(self) -> bool:
        return self.monthly_cost > 0

    @property
    def is_one_time(self) -> bool:
        return self.one_time_cost > 0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperationalExpense":
        return cls(
            name=str(payload.get("name", "")),
            monthly_cost=float(payload.get("monthly_cost", 0.0)),
            one_time_cost=float(payload.get("one_time_cost", 0.0)),
        )


@dataclass(slots=True)
class ElectricityProfile:
    name: str
    quantity: float
    max_power: float
    hours_per_day: float = 0.0
    working_days: float = 0.0
    round_the_clock: bool = False

    @property
    def energy_consumption(self) -> float:
        return (
            (float(self.max_power) / 1000.0)
            * float(self.quantity)
            * float(self.hours_per_day)
            * float(self.working_days)
        )

    def calculate_cost(self, cost_per_kwh: float) -> float:
        return self.energy_consumption * float(cost_per_kwh)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        hours_per_day: float = 0.0,
        working_days: float = 0.0,
        round_the_clock: bool = False,
    ) -> "ElectricityProfile":
        return cls(
            name=str(payload.get("name", "")),
            quantity=float(payload.get("quantity", 0.0)),
            max_power=float(payload.get("max_power", 0.0)),
            hours_per_day=float(payload.get("hours_per_day", hours_per_day)),
            working_days=float(payload.get("working_days", working_days)),
            round_the_clock=bool(payload.get("round_the_clock", round_the_clock)),
        )


@dataclass(slots=True)
class Criterion:
    id: str
    name: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Criterion":
        return cls(id=str(payload.get("id", "")), name=str(payload.get("name", "")))


@dataclass(slots=True)
class Alternative:
    id: str
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Alternative":
        metadata = {str(k): deepcopy(v) for k, v in payload.items() if k not in {"id", "name"}}
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", payload.get("id", ""))),
            metadata=metadata,
        )


@dataclass(slots=True)
class Relation:
    left: str
    operator: RelationOperator
    factor: float
    right: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Relation":
        return cls(
            left=str(payload.get("left", "")),
            operator=as_relation_operator(
                payload.get("op", payload.get("operator", RelationOperator.EQUAL.value))
            ),
            factor=float(payload.get("factor", 1.0)),
            right=str(payload.get("right", "")),
        )

    @property
    def op(self) -> str:
        return self.operator.value


@dataclass(slots=True)
class AnalysisResult:
    case_name: str
    ranking: list[dict[str, Any]] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    winner_id: str | None = None
    winner_name: str | None = None
    explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def as_equipment_category(value: str | EquipmentCategory) -> EquipmentCategory:
    if isinstance(value, EquipmentCategory):
        return value
    return EquipmentCategory(str(value))


def as_relation_operator(value: str | RelationOperator) -> RelationOperator:
    if isinstance(value, RelationOperator):
        return value
    return RelationOperator(str(value))


def to_plain_data(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: to_plain_data(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    return deepcopy(value)


T = TypeVar("T")


def ensure_capital_item(
    value: CapitalItem | Mapping[str, Any], *, category: str | EquipmentCategory | None = None
) -> CapitalItem:
    if isinstance(value, CapitalItem):
        if category is not None and value.category is None:
            return CapitalItem(
                name=value.name,
                quantity=value.quantity,
                price=value.price,
                category=as_equipment_category(category),
            )
        return value
    return CapitalItem.from_dict(value, category=category)


def ensure_operational_expense(value: OperationalExpense | Mapping[str, Any]) -> OperationalExpense:
    if isinstance(value, OperationalExpense):
        return value
    return OperationalExpense.from_dict(value)


def ensure_electricity_profile(
    value: ElectricityProfile | Mapping[str, Any],
    *,
    hours_per_day: float = 0.0,
    working_days: float = 0.0,
    round_the_clock: bool = False,
) -> ElectricityProfile:
    if isinstance(value, ElectricityProfile):
        return value
    return ElectricityProfile.from_dict(
        value,
        hours_per_day=hours_per_day,
        working_days=working_days,
        round_the_clock=round_the_clock,
    )


def ensure_criterion(value: Criterion | Mapping[str, Any]) -> Criterion:
    if isinstance(value, Criterion):
        return value
    return Criterion.from_dict(value)


def ensure_alternative(value: Alternative | Mapping[str, Any]) -> Alternative:
    if isinstance(value, Alternative):
        return value
    return Alternative.from_dict(value)


def ensure_relation(value: Relation | Mapping[str, Any]) -> Relation:
    if isinstance(value, Relation):
        return value
    return Relation.from_dict(value)
