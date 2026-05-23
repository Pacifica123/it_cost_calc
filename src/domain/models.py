from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, TypeVar


class ExpenseType(str, Enum):
    CAPITAL = "capital"
    OPERATIONAL = "operational"
    ELECTRICITY = "electricity"


class AnalysisScope(str, Enum):
    TECHNICAL = "technical"
    SOFTWARE = "software"
    IMPLEMENTATION = "implementation"
    COMMON = "common"


class CandidateConfigurationSource(str, Enum):
    MANUAL = "manual"
    DEMO = "demo"
    GA = "ga"
    CATALOG = "catalog"
    IMPORTED = "imported"


class ComponentType(str, Enum):
    SERVER = "server"
    WORKSTATION = "workstation"
    PERIPHERAL = "peripheral"
    NETWORK_DEVICE = "network_device"
    SOFTWARE_LICENSE = "software_license"
    SOFTWARE_SUBSCRIPTION = "software_subscription"
    SOFTWARE_SERVICE = "software_service"
    IMPLEMENTATION_SERVICE = "implementation_service"
    SUPPORT_SERVICE = "support_service"
    BACKUP_SERVICE = "backup_service"


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
    max_power: float | None = None
    client_seats: float | None = None
    scope: AnalysisScope | None = None
    component_type: ComponentType | None = None
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
            max_power=_optional_float(payload.get("max_power")),
            client_seats=_optional_float(payload.get("client_seats")),
            scope=as_analysis_scope(payload["scope"]) if payload.get("scope") else None,
            component_type=(
                as_component_type(payload["component_type"])
                if payload.get("component_type")
                else None
            ),
        )


@dataclass(slots=True)
class OperationalExpense:
    name: str
    monthly_cost: float = 0.0
    one_time_cost: float = 0.0
    scope: AnalysisScope | None = None
    component_type: ComponentType | None = None
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
            scope=as_analysis_scope(payload["scope"]) if payload.get("scope") else None,
            component_type=(
                as_component_type(payload["component_type"])
                if payload.get("component_type")
                else None
            ),
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
class CostModel:
    """Cost breakdown for one component or a whole candidate configuration.

    The model keeps one-time investment costs separate from recurring monthly
    costs.  This lets application services build TCO summaries and NPV cash
    flows without guessing which value came from CAPEX, OPEX or electricity.
    """

    purchase_cost: float = 0.0
    implementation_cost: float = 0.0
    testing_cost: float = 0.0
    migration_cost: float = 0.0
    subscription_cost: float = 0.0
    support_cost: float = 0.0
    electricity_cost: float = 0.0

    @property
    def one_time_costs(self) -> float:
        return (
            float(self.purchase_cost)
            + float(self.implementation_cost)
            + float(self.testing_cost)
            + float(self.migration_cost)
        )

    @property
    def monthly_costs(self) -> float:
        return (
            float(self.subscription_cost)
            + float(self.support_cost)
            + float(self.electricity_cost)
        )

    @property
    def annual_costs(self) -> float:
        return self.monthly_costs * 12.0

    def total_for_months(self, horizon_months: int | float) -> float:
        return self.one_time_costs + self.monthly_costs * float(horizon_months)

    def plus(self, other: "CostModel") -> "CostModel":
        return CostModel(
            purchase_cost=self.purchase_cost + other.purchase_cost,
            implementation_cost=self.implementation_cost + other.implementation_cost,
            testing_cost=self.testing_cost + other.testing_cost,
            migration_cost=self.migration_cost + other.migration_cost,
            subscription_cost=self.subscription_cost + other.subscription_cost,
            support_cost=self.support_cost + other.support_cost,
            electricity_cost=self.electricity_cost + other.electricity_cost,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CostModel":
        return cls(
            purchase_cost=float(payload.get("purchase_cost", 0.0)),
            implementation_cost=float(payload.get("implementation_cost", 0.0)),
            testing_cost=float(payload.get("testing_cost", 0.0)),
            migration_cost=float(payload.get("migration_cost", 0.0)),
            subscription_cost=float(payload.get("subscription_cost", 0.0)),
            support_cost=float(payload.get("support_cost", 0.0)),
            electricity_cost=float(payload.get("electricity_cost", 0.0)),
        )


@dataclass(slots=True)
class CandidateConfiguration:
    """Common candidate alternative format for AHP, GA and reports.

    The model separates original components from aggregated totals and metrics.
    It is intentionally permissive: old AHP dictionaries and GA solutions can be
    adapted into this shape without forcing their legacy fields to disappear.
    """

    id: str
    name: str
    scope: AnalysisScope | None = None
    components: list[dict[str, Any]] = field(default_factory=list)
    totals: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    source: CandidateConfigurationSource = CandidateConfigurationSource.MANUAL
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateConfiguration":
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", payload.get("id", ""))),
            scope=as_analysis_scope(payload["scope"]) if payload.get("scope") else None,
            components=[
                {str(key): deepcopy(value) for key, value in dict(component).items()}
                for component in payload.get("components", [])
                if isinstance(component, Mapping)
            ],
            totals={str(key): deepcopy(value) for key, value in dict(payload.get("totals", {})).items()},
            metrics={str(key): deepcopy(value) for key, value in dict(payload.get("metrics", {})).items()},
            source=as_candidate_configuration_source(payload.get("source", CandidateConfigurationSource.MANUAL.value)),
            metadata={
                str(key): deepcopy(value)
                for key, value in dict(payload.get("metadata", {})).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return to_plain_data(self)


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


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def as_analysis_scope(value: str | AnalysisScope) -> AnalysisScope:
    if isinstance(value, AnalysisScope):
        return value
    return AnalysisScope(str(value))


def as_candidate_configuration_source(
    value: str | CandidateConfigurationSource,
) -> CandidateConfigurationSource:
    if isinstance(value, CandidateConfigurationSource):
        return value
    return CandidateConfigurationSource(str(value))


def as_component_type(value: str | ComponentType) -> ComponentType:
    if isinstance(value, ComponentType):
        return value
    return ComponentType(str(value))


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
                max_power=value.max_power,
                client_seats=value.client_seats,
                scope=value.scope,
                component_type=value.component_type,
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


def ensure_candidate_configuration(
    value: CandidateConfiguration | Mapping[str, Any],
) -> CandidateConfiguration:
    if isinstance(value, CandidateConfiguration):
        return value
    return CandidateConfiguration.from_dict(value)

def ensure_cost_model(value: CostModel | Mapping[str, Any]) -> CostModel:
    if isinstance(value, CostModel):
        return value
    return CostModel.from_dict(value)


def ensure_relation(value: Relation | Mapping[str, Any]) -> Relation:
    if isinstance(value, Relation):
        return value
    return Relation.from_dict(value)
