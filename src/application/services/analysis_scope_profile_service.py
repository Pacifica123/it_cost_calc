"""Application-level profiles for analysis scopes.

The project has separate UI tabs for technical and software analysis, but the
rules that make those scopes different must live outside widgets.  This module
keeps the transition profile explicit: UI can ask for labels, categories,
criteria, constraints and explanatory text, while GA/AHP orchestration receives
plain callables and serializable metadata.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from domain import AnalysisScope, ComponentType
from shared.constants import (
    ANALYSIS_SCOPE_SOFTWARE,
    ANALYSIS_SCOPE_TECHNICAL,
    OPERATIONAL_COST_CATEGORIES,
    SOFTWARE_CAPITAL_CATEGORIES,
    TECHNICAL_CAPITAL_CATEGORIES,
)


@dataclass(frozen=True, slots=True)
class AnalysisCriterionProfile:
    """Descriptor of one criterion inside an analysis-scope profile."""

    id: str
    label: str
    direction: str = "max"
    metric: str | None = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class AnalysisConstraintProfile:
    """Descriptor of a profile-level restriction.

    The descriptor is intentionally data-only.  Runtime values such as a concrete
    budget or power limit are supplied by use cases or UI forms at execution time.
    """

    id: str
    label: str
    kind: str
    required: bool = False
    category: str | None = None
    metric: str | None = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class AnalysisScopeProfile:
    """Serializable contract for one analysis scope."""

    scope: str
    label: str
    title: str
    capital_categories: tuple[str, ...]
    operational_categories: tuple[str, ...] = ()
    component_types: tuple[str, ...] = ()
    criteria: tuple[AnalysisCriterionProfile, ...] = ()
    constraints: tuple[AnalysisConstraintProfile, ...] = ()
    default_weights: Mapping[str, float] = field(default_factory=dict)
    metric_extractors: Mapping[str, str] = field(default_factory=dict)
    explanation_rules: Mapping[str, str] = field(default_factory=dict)

    def criterion_ids(self) -> tuple[str, ...]:
        return tuple(criterion.id for criterion in self.criteria)

    def constraint_ids(self) -> tuple[str, ...]:
        return tuple(constraint.id for constraint in self.constraints)

    def default_required_categories(self) -> tuple[str, ...]:
        return tuple(
            constraint.category
            for constraint in self.constraints
            if constraint.kind == "required_category" and constraint.required and constraint.category
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["capital_categories"] = list(self.capital_categories)
        payload["operational_categories"] = list(self.operational_categories)
        payload["component_types"] = list(self.component_types)
        payload["criteria"] = [asdict(criterion) for criterion in self.criteria]
        payload["constraints"] = [asdict(constraint) for constraint in self.constraints]
        payload["default_weights"] = dict(self.default_weights)
        payload["metric_extractors"] = dict(self.metric_extractors)
        payload["explanation_rules"] = dict(self.explanation_rules)
        return payload


class AnalysisScopeProfileService:
    """Registry and helper methods for ПО/ТО analysis profiles."""

    def __init__(self, profiles: Sequence[AnalysisScopeProfile] | None = None):
        profile_items = tuple(profiles or DEFAULT_ANALYSIS_SCOPE_PROFILES)
        self._profiles = {profile.scope: profile for profile in profile_items}

    def get_profile(self, scope: str | None) -> AnalysisScopeProfile:
        scope_key = str(scope or ANALYSIS_SCOPE_TECHNICAL)
        if scope_key not in self._profiles:
            return self._profiles[ANALYSIS_SCOPE_TECHNICAL]
        return self._profiles[scope_key]

    def profiles(self) -> dict[str, AnalysisScopeProfile]:
        return dict(self._profiles)

    def labels(self) -> dict[str, str]:
        return {scope: profile.label for scope, profile in self._profiles.items()}

    def titles(self) -> dict[str, str]:
        return {scope: profile.title for scope, profile in self._profiles.items()}

    def default_weights(self, scope: str | None) -> list[float]:
        profile = self.get_profile(scope)
        return [float(profile.default_weights.get(criterion.id, 1.0)) for criterion in profile.criteria]

    def profile_metadata(self, scope: str | None) -> dict[str, Any]:
        return deepcopy(self.get_profile(scope).to_dict())

    def build_ga_criteria(
        self,
        scope: str | None,
        *,
        power_lookup: Mapping[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Build GA criterion callables from a profile.

        The returned structure remains compatible with the generic GA core, but
        the choice of criteria is no longer embedded in a Tkinter tab.
        """

        profile = self.get_profile(scope)
        lookup = dict(power_lookup or {})
        result: list[dict[str, Any]] = []
        for criterion in profile.criteria:
            result.append(
                {
                    "name": criterion.id,
                    "label": criterion.label,
                    "direction": criterion.direction,
                    "func": self._metric_callable(criterion.metric or criterion.id, lookup),
                    "profile_scope": profile.scope,
                }
            )
        return result

    def build_ga_constraints(
        self,
        scope: str | None,
        *,
        power_lookup: Mapping[str, float] | None = None,
        max_power: float | None = None,
        target_units: float | None = None,
    ) -> list[dict[str, Any]]:
        """Build dynamic GA constraints described by the profile."""

        profile = self.get_profile(scope)
        lookup = dict(power_lookup or {})
        result: list[dict[str, Any]] = []
        for descriptor in profile.constraints:
            if descriptor.kind != "metric_minimum":
                continue
            if descriptor.metric == "total_power_watts" and max_power is not None:
                result.append(
                    {
                        "name": descriptor.id,
                        "label": descriptor.label,
                        "func": self._metric_callable("total_power_watts", lookup),
                        "operator": "<=",
                        "bound": float(max_power),
                        "profile_scope": profile.scope,
                    }
                )
            elif descriptor.metric in {"client_capacity", "software_license_quantity"} and target_units:
                result.append(
                    {
                        "name": descriptor.id,
                        "label": descriptor.label,
                        "func": self._metric_callable(descriptor.metric, lookup),
                        "operator": ">=",
                        "bound": float(target_units),
                        "profile_scope": profile.scope,
                    }
                )
        return result

    def required_categories(
        self,
        scope: str | None,
        *,
        enabled_categories: Mapping[str, bool] | None = None,
    ) -> list[str]:
        profile = self.get_profile(scope)
        enabled = dict(enabled_categories or {})
        result: list[str] = []
        for descriptor in profile.constraints:
            if descriptor.kind != "required_category" or not descriptor.category:
                continue
            if enabled.get(descriptor.category, descriptor.required):
                result.append(descriptor.category)
        return result

    def criterion_label(self, criterion_id: str, *, scope: str | None = None) -> str:
        profile = self.get_profile(scope)
        for criterion in profile.criteria:
            if criterion.id == criterion_id:
                return criterion.label
        for fallback in self._profiles.values():
            for criterion in fallback.criteria:
                if criterion.id == criterion_id:
                    return criterion.label
        return criterion_id

    def constraint_label(self, constraint_id: str, *, scope: str | None = None) -> str:
        profile = self.get_profile(scope)
        for constraint in profile.constraints:
            if constraint.id == constraint_id:
                return constraint.label
        for fallback in self._profiles.values():
            for constraint in fallback.constraints:
                if constraint.id == constraint_id:
                    return constraint.label
        return constraint_id

    def _metric_callable(self, metric: str, power_lookup: Mapping[str, float]):
        if metric == "category_coverage":
            return self._category_coverage
        if metric == "client_capacity":
            return self._client_capacity
        if metric == "software_license_quantity":
            return self._software_quantity
        if metric == "total_power_watts":
            return lambda subset, lookup=power_lookup: self._total_power(subset, lookup)
        if metric in {"selected_software_items", "selected_count"}:
            return lambda subset: float(len(subset))
        if metric in {"capital_cost", "total_cost", "ownership_cost"}:
            return lambda subset: self._sum_property(subset, "total_cost")
        if metric == "support_score":
            return lambda subset: self._average_property(subset, "support_score")
        if metric == "functionality_score":
            return lambda subset: self._average_property(subset, "functionality_score")
        return lambda subset: self._sum_property(subset, metric)

    def _category_coverage(self, subset: Sequence[Any]) -> float:
        return float(
            len(
                {
                    str(self._properties(item).get("source_category") or self._properties(item).get("category"))
                    for item in subset
                    if self._properties(item).get("source_category") or self._properties(item).get("category")
                }
            )
        )

    def _client_capacity(self, subset: Sequence[Any]) -> float:
        return sum(self._item_client_capacity(self._properties(item)) for item in subset)

    def _software_quantity(self, subset: Sequence[Any]) -> float:
        return sum(self._item_software_quantity(self._properties(item)) for item in subset)

    def _total_power(self, subset: Sequence[Any], power_lookup: Mapping[str, float]) -> float:
        return sum(self._item_power(self._properties(item), power_lookup) for item in subset)

    def _sum_property(self, subset: Sequence[Any], property_name: str) -> float:
        return sum(self._number(self._properties(item).get(property_name), default=0.0) for item in subset)

    def _average_property(self, subset: Sequence[Any], property_name: str) -> float:
        values = [self._number(self._properties(item).get(property_name), default=0.0) for item in subset]
        return sum(values) / len(values) if values else 0.0

    def _properties(self, item: Any) -> Mapping[str, Any]:
        if isinstance(item, Mapping):
            return item
        properties = getattr(item, "properties", None)
        if isinstance(properties, Mapping):
            return properties
        return {}

    def _item_software_quantity(self, item: Mapping[str, Any]) -> float:
        if "license_units" in item:
            return self._number(item.get("license_units"), default=1.0)
        return self._number(item.get("quantity"), default=1.0)

    def _item_client_capacity(self, item: Mapping[str, Any]) -> float:
        if "client_seats" in item:
            return self._number(item.get("client_seats"), default=0.0)
        if item.get("component_type") == ComponentType.WORKSTATION.value:
            return self._number(item.get("quantity"), default=0.0)
        return 0.0

    def _item_power(self, item: Mapping[str, Any], power_lookup: Mapping[str, float]) -> float:
        name = str(item.get("name", ""))
        quantity = self._number(item.get("quantity"), default=1.0)
        per_unit_power = self._number(item.get("max_power", power_lookup.get(name, 0.0)), default=0.0)
        return quantity * per_unit_power

    def _number(self, value: Any, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)


TECHNICAL_PROFILE = AnalysisScopeProfile(
    scope=AnalysisScope.TECHNICAL.value,
    label="ТО",
    title="Техническое обеспечение",
    capital_categories=TECHNICAL_CAPITAL_CATEGORIES,
    operational_categories=("server_rental", "backup", "server_administration"),
    component_types=(
        ComponentType.SERVER.value,
        ComponentType.WORKSTATION.value,
        ComponentType.PERIPHERAL.value,
        ComponentType.NETWORK_DEVICE.value,
        ComponentType.SUPPORT_SERVICE.value,
        ComponentType.BACKUP_SERVICE.value,
    ),
    criteria=(
        AnalysisCriterionProfile(
            id="category_coverage",
            label="Покрытие технических контуров",
            direction="max",
            metric="category_coverage",
            description="Наличие серверного, клиентского и сетевого контура.",
        ),
        AnalysisCriterionProfile(
            id="client_capacity",
            label="Покрытие рабочих мест",
            direction="max",
            metric="client_capacity",
            description="Количество рабочих мест по client_seats; периферия не занимает место пользователя.",
        ),
        AnalysisCriterionProfile(
            id="total_power_watts",
            label="Энергопотребление",
            direction="min",
            metric="total_power_watts",
            description="Суммарная мощность выбранных технических компонентов.",
        ),
        AnalysisCriterionProfile(
            id="capital_cost",
            label="Стоимость",
            direction="min",
            metric="capital_cost",
            description="Капитальная стоимость выбранного набора.",
        ),
    ),
    constraints=(
        AnalysisConstraintProfile(
            id="budget",
            label="Бюджет",
            kind="budget_limit",
            metric="capital_cost",
            description="Универсальный лимит стоимости, который передаётся в GA как max_budget.",
        ),
        AnalysisConstraintProfile(
            id="required_category_server",
            label="Обязательный серверный контур",
            kind="required_category",
            required=True,
            category="server",
        ),
        AnalysisConstraintProfile(
            id="required_category_client",
            label="Обязательный клиентский контур",
            kind="required_category",
            required=True,
            category="client",
        ),
        AnalysisConstraintProfile(
            id="required_category_network",
            label="Обязательный сетевой контур",
            kind="required_category",
            required=True,
            category="network",
        ),
        AnalysisConstraintProfile(
            id="power_limit",
            label="Лимит мощности",
            kind="metric_minimum",
            metric="total_power_watts",
        ),
        AnalysisConstraintProfile(
            id="client_capacity_required",
            label="Минимум клиентских мест",
            kind="metric_minimum",
            metric="client_capacity",
        ),
    ),
    default_weights={
        "category_coverage": 0.35,
        "client_capacity": 0.25,
        "total_power_watts": 0.15,
        "capital_cost": 0.25,
    },
    metric_extractors={
        "category_coverage": "count unique technical capital categories",
        "client_capacity": "sum client_seats, workstation fallback only",
        "total_power_watts": "quantity × max_power with energy-tab lookup fallback",
        "capital_cost": "sum total_cost",
    },
    explanation_rules={
        "ui_hint": (
            "Режим ТО: анализируются серверы, клиентские устройства и сеть. "
            "Клиентская ёмкость считается по client_seats, поэтому МФУ не заменяет рабочее место."
        ),
        "winner_template": "Выбрана конфигурация ТО с учётом технических контуров, мощности и рабочих мест.",
    },
)


SOFTWARE_PROFILE = AnalysisScopeProfile(
    scope=AnalysisScope.SOFTWARE.value,
    label="ПО",
    title="Программное обеспечение",
    capital_categories=SOFTWARE_CAPITAL_CATEGORIES,
    operational_categories=("subscription_licenses", "migration", "testing", "labor_costs"),
    component_types=(
        ComponentType.SOFTWARE_LICENSE.value,
        ComponentType.SOFTWARE_SUBSCRIPTION.value,
        ComponentType.SOFTWARE_SERVICE.value,
        ComponentType.IMPLEMENTATION_SERVICE.value,
        ComponentType.SUPPORT_SERVICE.value,
    ),
    criteria=(
        AnalysisCriterionProfile(
            id="selected_software_items",
            label="Разнообразие ПО",
            direction="max",
            metric="selected_software_items",
            description="Количество выбранных программных компонентов.",
        ),
        AnalysisCriterionProfile(
            id="software_license_quantity",
            label="Количество лицензий или пользователей",
            direction="max",
            metric="software_license_quantity",
            description="Лицензионная ёмкость без подмены клиентскими устройствами.",
        ),
        AnalysisCriterionProfile(
            id="capital_cost",
            label="Стоимость владения ПО",
            direction="min",
            metric="capital_cost",
            description="Стоимость лицензий и программных компонентов в выбранном наборе.",
        ),
    ),
    constraints=(
        AnalysisConstraintProfile(
            id="budget",
            label="Бюджет",
            kind="budget_limit",
            metric="capital_cost",
            description="Универсальный лимит стоимости, который передаётся в GA как max_budget.",
        ),
        AnalysisConstraintProfile(
            id="required_category_licenses",
            label="Обязательный лицензионный контур",
            kind="required_category",
            required=True,
            category="licenses",
        ),
        AnalysisConstraintProfile(
            id="software_license_quantity_required",
            label="Минимум лицензий",
            kind="metric_minimum",
            metric="software_license_quantity",
        ),
        AnalysisConstraintProfile(
            id="subscription_monthly_limit",
            label="Лимит ежемесячной подписки",
            kind="reserved_tco_limit",
            metric="monthly_subscription_cost",
            description="Зарезервировано для развития OPEX/TCO-профиля ПО.",
        ),
        AnalysisConstraintProfile(
            id="functional_coverage_minimum",
            label="Минимальное функциональное покрытие",
            kind="reserved_quality_minimum",
            metric="functionality_score",
            description="Зарезервировано до появления явной функциональной модели альтернатив.",
        ),
        AnalysisConstraintProfile(
            id="support_required",
            label="Наличие сопровождения",
            kind="reserved_support_presence",
            metric="support_score",
            description="Зарезервировано для связи программного профиля с OPEX-сопровождением.",
        ),
    ),
    default_weights={
        "selected_software_items": 0.35,
        "software_license_quantity": 0.30,
        "capital_cost": 0.35,
    },
    metric_extractors={
        "selected_software_items": "count selected software rows",
        "software_license_quantity": "sum quantity/license_units",
        "capital_cost": "sum total_cost",
    },
    explanation_rules={
        "ui_hint": (
            "Режим ПО: анализируются лицензии и программные сервисы. Мощность не является "
            "обязательным критерием; поле минимума трактуется как минимум лицензий/пользователей."
        ),
        "winner_template": "Выбран набор ПО без технических ограничений по серверу, сети и мощности.",
    },
)


IT_SOLUTION_PROFILE = AnalysisScopeProfile(
    scope="solution",
    label="ИТ-решение",
    title="Сводный профиль всего ИТ-решения",
    capital_categories=TECHNICAL_CAPITAL_CATEGORIES + SOFTWARE_CAPITAL_CATEGORIES,
    operational_categories=OPERATIONAL_COST_CATEGORIES,
    component_types=TECHNICAL_PROFILE.component_types + SOFTWARE_PROFILE.component_types,
    criteria=TECHNICAL_PROFILE.criteria + SOFTWARE_PROFILE.criteria,
    constraints=TECHNICAL_PROFILE.constraints + SOFTWARE_PROFILE.constraints,
    default_weights={**TECHNICAL_PROFILE.default_weights, **SOFTWARE_PROFILE.default_weights},
    metric_extractors={**TECHNICAL_PROFILE.metric_extractors, **SOFTWARE_PROFILE.metric_extractors},
    explanation_rules={
        "ui_hint": "Будущий сводный профиль объединит ТО, ПО, внедрение и общие расходы.",
        "status": "reserved_for_stage4_plus",
    },
)


DEFAULT_ANALYSIS_SCOPE_PROFILES = (TECHNICAL_PROFILE, SOFTWARE_PROFILE)

__all__ = [
    "AnalysisConstraintProfile",
    "AnalysisCriterionProfile",
    "AnalysisScopeProfile",
    "AnalysisScopeProfileService",
    "DEFAULT_ANALYSIS_SCOPE_PROFILES",
    "IT_SOLUTION_PROFILE",
    "SOFTWARE_PROFILE",
    "TECHNICAL_PROFILE",
]
