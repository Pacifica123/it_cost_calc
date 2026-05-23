from __future__ import annotations

from copy import deepcopy
from statistics import median
from typing import Any, Iterable, Mapping

from application.services.analysis_scope_profile_service import AnalysisScopeProfileService
from application.services.runtime_entity_normalization_service import normalize_runtime_row
from domain.decision.ahp.aggregation import aggregate_configuration
from shared.constants import (
    ANALYSIS_SCOPE_SOFTWARE,
    ANALYSIS_SCOPE_TECHNICAL,
    OPERATIONAL_COST_CATEGORIES,
)

CRITERIA_IMPORTANCE_CRITERIA = [
    {"id": "1", "name": "Надежность и отказоустойчивость"},
    {"id": "2", "name": "Производительность конфигурации"},
    {"id": "3", "name": "Стоимость владения"},
    {"id": "4", "name": "Энергоэффективность"},
    {"id": "5", "name": "Потенциал развития и срок службы"},
]

SOFTWARE_CRITERIA_IMPORTANCE_CRITERIA = [
    {"id": "1", "name": "Функциональное покрытие"},
    {"id": "2", "name": "Лицензионная обеспеченность пользователей"},
    {"id": "3", "name": "Стоимость владения ПО"},
    {"id": "4", "name": "Простота сопровождения"},
    {"id": "5", "name": "Потенциал развития и совместимость"},
]

CRITERIA_IMPORTANCE_RELATIONS = [
    {"left": "1", "op": ">", "factor": 2.0, "right": "3"},
    {"left": "1", "op": ">", "factor": 2.0, "right": "4"},
    {"left": "1", "op": "=", "factor": 1.0, "right": "2"},
    {"left": "2", "op": ">", "factor": 1.5, "right": "3"},
    {"left": "2", "op": ">", "factor": 1.5, "right": "4"},
    {"left": "5", "op": "=", "factor": 1.0, "right": "2"},
    {"left": "5", "op": ">", "factor": 4.0 / 3.0, "right": "3"},
    {"left": "3", "op": ">", "factor": 4.0 / 3.0, "right": "4"},
]

SOFTWARE_CRITERIA_IMPORTANCE_RELATIONS = [
    {"left": "1", "op": ">", "factor": 2.0, "right": "3"},
    {"left": "2", "op": ">", "factor": 1.5, "right": "3"},
    {"left": "1", "op": "=", "factor": 1.0, "right": "5"},
    {"left": "5", "op": ">", "factor": 1.5, "right": "4"},
    {"left": "4", "op": ">", "factor": 4.0 / 3.0, "right": "3"},
]

_ROLE_PROFILES = {
    "server": {
        "cpu": (5.0, 8.8),
        "ram": (4.0, 7.2),
        "energy": (240.0, 420.0),
        "rel_low": (0.82, 0.93),
        "rel_high": (0.92, 0.985),
        "lifespan": (5.0, 8.0),
    },
    "client": {
        "cpu": (1.6, 4.2),
        "ram": (1.0, 3.2),
        "energy": (45.0, 105.0),
        "rel_low": (0.72, 0.88),
        "rel_high": (0.84, 0.96),
        "lifespan": (3.0, 5.5),
    },
    "network": {
        "cpu": (0.3, 1.4),
        "ram": (0.2, 1.0),
        "energy": (18.0, 65.0),
        "rel_low": (0.78, 0.9),
        "rel_high": (0.9, 0.98),
        "lifespan": (4.0, 7.0),
    },
    "software": {
        "functionality": (2.0, 5.0),
        "support": (2.4, 4.8),
        "rel_low": (0.76, 0.9),
        "rel_high": (0.86, 0.98),
        "lifespan": (2.0, 5.0),
    },
}


class DecisionDemoDataService:
    """Builds consistent demo data for AHP and criteria-importance tabs from runtime entities."""

    def __init__(self, profile_service: AnalysisScopeProfileService | None = None):
        self.profile_service = profile_service or AnalysisScopeProfileService()

    def build(
        self,
        entities: Mapping[str, list[dict[str, Any]]],
        *,
        analysis_scope: str = ANALYSIS_SCOPE_TECHNICAL,
    ) -> dict[str, Any]:
        scoped_payloads = {
            scope: self.build_scope_payload(entities, analysis_scope=scope)
            for scope in self.profile_service.profiles()
            if scope in {ANALYSIS_SCOPE_TECHNICAL, ANALYSIS_SCOPE_SOFTWARE}
        }
        selected = scoped_payloads.get(analysis_scope, scoped_payloads[ANALYSIS_SCOPE_TECHNICAL])
        result = deepcopy(selected)
        result["scoped_payloads"] = scoped_payloads
        return result

    def build_scope_payload(
        self,
        entities: Mapping[str, list[dict[str, Any]]],
        *,
        analysis_scope: str,
    ) -> dict[str, Any]:
        configurations = self.build_ahp_configurations(entities, analysis_scope=analysis_scope)
        constraints = self.build_recommended_constraints(
            configurations, analysis_scope=analysis_scope
        )
        criteria_case = self.build_criteria_importance_case(
            configurations, analysis_scope=analysis_scope
        )
        return {
            "analysis_scope": analysis_scope,
            "analysis_profile": self.profile_service.profile_metadata(analysis_scope),
            "configurations": configurations,
            "constraints": constraints,
            "criteria_case": criteria_case,
        }

    def build_ahp_configurations(
        self,
        entities: Mapping[str, list[dict[str, Any]]],
        *,
        analysis_scope: str = ANALYSIS_SCOPE_TECHNICAL,
    ) -> list[dict[str, Any]]:
        if analysis_scope == ANALYSIS_SCOPE_SOFTWARE:
            return self._build_software_configurations(entities)
        return self._build_technical_configurations(entities)

    def _build_technical_configurations(
        self, entities: Mapping[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        servers = self._expand_category(entities.get("server", []), role="server")
        clients = self._expand_category(entities.get("client", []), role="client")
        networks = self._expand_category(entities.get("network", []), role="network")

        if not servers or not clients:
            raise ValueError(
                "Для подготовки демонстрационных конфигураций ТО нужны серверы и клиентские устройства"
            )

        client_target = max(2, self._client_seat_capacity(clients) or len(clients))
        configs = [
            self._make_config(
                config_id="to_budget",
                name="ТО: экономичная конфигурация",
                people=max(2, client_target - 1),
                devices=[
                    *self._pick_units(servers, 1, "low"),
                    *self._pick_units(clients, max(2, client_target - 1), "low"),
                    *self._pick_units(networks, 1, "low"),
                ],
            ),
            self._make_config(
                config_id="to_compact",
                name="ТО: компактная конфигурация",
                people=client_target,
                devices=[
                    *self._pick_units(servers, 1, "mid"),
                    *self._pick_units(clients, client_target, "low_mid"),
                    *self._pick_units(networks, 1, "mid"),
                ],
            ),
            self._make_config(
                config_id="to_balanced",
                name="ТО: сбалансированная конфигурация",
                people=client_target,
                devices=[
                    *self._pick_units(servers, 1, "mid"),
                    *self._pick_units(clients, client_target, "mid"),
                    *self._pick_units(networks, min(2, max(len(networks), 1)), "mid"),
                ],
            ),
            self._make_config(
                config_id="to_growth",
                name="ТО: конфигурация с запасом роста",
                people=client_target,
                devices=[
                    *self._pick_units(servers, min(2, len(servers)), "high"),
                    *self._pick_units(clients, client_target, "high"),
                    *self._pick_units(networks, min(2, max(len(networks), 1)), "high"),
                ],
            ),
            self._make_config(
                config_id="to_reserve",
                name="ТО: резервоустойчивая конфигурация",
                people=client_target,
                devices=[
                    *self._pick_units(servers, max(1, min(2, len(servers))), "high"),
                    *self._pick_units(clients, client_target + 1, "mid_high"),
                    *self._pick_units(networks, max(1, min(2, len(networks))), "high"),
                ],
            ),
        ]
        return configs

    def _build_software_configurations(
        self, entities: Mapping[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        licenses = self._expand_software_rows(entities)
        if not licenses:
            raise ValueError("Для подготовки демонстрационных конфигураций ПО нужны лицензии")

        license_target = max(1, min(5, self._license_units(licenses) or len(licenses)))
        configs = [
            self._make_config(
                config_id="po_basic",
                name="ПО: базовый набор лицензий",
                people=0,
                devices=self._pick_units(licenses, max(1, license_target - 1), "low"),
            ),
            self._make_config(
                config_id="po_user_pack",
                name="ПО: пользовательский пакет",
                people=0,
                devices=self._pick_units(licenses, license_target, "low_mid"),
            ),
            self._make_config(
                config_id="po_balanced",
                name="ПО: сбалансированный набор",
                people=0,
                devices=self._pick_units(licenses, license_target + 1, "mid"),
            ),
            self._make_config(
                config_id="po_extended",
                name="ПО: расширенный функциональный набор",
                people=0,
                devices=self._pick_units(licenses, license_target + 2, "mid_high"),
            ),
            self._make_config(
                config_id="po_growth",
                name="ПО: набор с запасом развития",
                people=0,
                devices=self._pick_units(licenses, license_target + 3, "high"),
            ),
        ]
        return configs

    def build_recommended_constraints(
        self,
        configurations: Iterable[dict[str, Any]],
        *,
        analysis_scope: str = ANALYSIS_SCOPE_TECHNICAL,
    ) -> dict[str, float]:
        aggregates = [aggregate_configuration(config) for config in configurations]
        costs = sorted(float(item["total_cost"]) for item in aggregates)
        energies = sorted(float(item["total_energy"]) for item in aggregates)
        if not costs or not energies:
            raise ValueError("Не удалось вычислить ограничения для пустого набора конфигураций")

        pivot_index = min(len(costs) - 1, 3)
        max_budget = round(costs[pivot_index] * 1.05, 2)
        max_energy = round(max(energies[pivot_index] * 1.08, 0.0), 2)
        result: dict[str, float] = {
            "max_budget": max_budget,
            "max_energy": max_energy,
        }
        if analysis_scope == ANALYSIS_SCOPE_TECHNICAL:
            result["people_match_tolerance"] = 0.25
        else:
            result["people_match_tolerance"] = 1.0
        return result

    def build_criteria_importance_case(
        self,
        configurations: Iterable[dict[str, Any]],
        *,
        analysis_scope: str = ANALYSIS_SCOPE_TECHNICAL,
    ) -> dict[str, Any]:
        if analysis_scope == ANALYSIS_SCOPE_SOFTWARE:
            return self._build_software_criteria_importance_case(configurations)
        return self._build_technical_criteria_importance_case(configurations)

    def _build_technical_criteria_importance_case(
        self, configurations: Iterable[dict[str, Any]]
    ) -> dict[str, Any]:
        configs = list(configurations)
        aggregates = {config["id"]: aggregate_configuration(config) for config in configs}
        alternatives = [{"id": config["id"], "name": config.get("name", config["id"])} for config in configs]

        scores_map = {
            "1": self._scaled_scores([aggregates[alt["id"]]["avg_reliability"] for alt in alternatives]),
            "2": self._scaled_scores([aggregates[alt["id"]]["total_performance"] for alt in alternatives]),
            "3": self._scaled_scores(
                [aggregates[alt["id"]]["total_cost"] for alt in alternatives], reverse=True
            ),
            "4": self._scaled_scores(
                [aggregates[alt["id"]]["total_energy"] for alt in alternatives], reverse=True
            ),
            "5": self._scaled_scores([aggregates[alt["id"]]["lifespan"] for alt in alternatives]),
        }

        scores = self._scores_by_alternative(alternatives, scores_map)
        return {
            "name": "ТО: выбор конфигурации технического обеспечения",
            "analysis_scope": ANALYSIS_SCOPE_TECHNICAL,
            "criteria": deepcopy(CRITERIA_IMPORTANCE_CRITERIA),
            "alternatives": alternatives,
            "scores": scores,
            "relations": deepcopy(CRITERIA_IMPORTANCE_RELATIONS),
        }

    def _build_software_criteria_importance_case(
        self, configurations: Iterable[dict[str, Any]]
    ) -> dict[str, Any]:
        configs = list(configurations)
        aggregates = {config["id"]: aggregate_configuration(config) for config in configs}
        alternatives = [{"id": config["id"], "name": config.get("name", config["id"])} for config in configs]
        license_units = [self._license_units(config.get("devices", [])) for config in configs]
        support_scores = [self._software_support_score(config.get("devices", [])) for config in configs]
        functionality_scores = [aggregates[alt["id"]]["total_performance"] for alt in alternatives]

        scores_map = {
            "1": self._scaled_scores(functionality_scores),
            "2": self._scaled_scores(license_units),
            "3": self._scaled_scores(
                [aggregates[alt["id"]]["total_cost"] for alt in alternatives], reverse=True
            ),
            "4": self._scaled_scores(support_scores),
            "5": self._scaled_scores([aggregates[alt["id"]]["lifespan"] for alt in alternatives]),
        }
        scores = self._scores_by_alternative(alternatives, scores_map)
        return {
            "name": "ПО: выбор набора программного обеспечения",
            "analysis_scope": ANALYSIS_SCOPE_SOFTWARE,
            "criteria": deepcopy(SOFTWARE_CRITERIA_IMPORTANCE_CRITERIA),
            "alternatives": alternatives,
            "scores": scores,
            "relations": deepcopy(SOFTWARE_CRITERIA_IMPORTANCE_RELATIONS),
        }

    def _scores_by_alternative(
        self,
        alternatives: list[dict[str, str]],
        scores_map: dict[str, list[float]],
    ) -> dict[str, dict[str, float]]:
        scores: dict[str, dict[str, float]] = {}
        for idx, alt in enumerate(alternatives):
            scores[alt["id"]] = {
                criterion_id: scores_values[idx]
                for criterion_id, scores_values in scores_map.items()
            }
        return scores

    def _expand_category(
        self, rows: Iterable[Mapping[str, Any]], *, role: str
    ) -> list[dict[str, Any]]:
        prepared_rows = [
            normalize_runtime_row(row, category=role)
            for row in rows
            if float(row.get("price", 0.0) or 0.0) > 0
        ]
        if not prepared_rows:
            return []

        prices = [float(row.get("price", 0.0) or 0.0) for row in prepared_rows]
        min_price = min(prices)
        max_price = max(prices)
        span = max(max_price - min_price, 1.0)

        expanded: list[dict[str, Any]] = []
        for row in prepared_rows:
            quantity = max(1, int(row.get("quantity", 0) or 0))
            norm = (float(row.get("price", 0.0) or 0.0) - min_price) / span if span else 0.5
            template = self._device_template(row, role=role, quality=float(norm))
            for _ in range(quantity):
                expanded.append(deepcopy(template))

        expanded.sort(key=lambda item: (float(item.get("cost", 0.0)), str(item.get("source_name", ""))))
        return expanded

    def _expand_software_rows(self, entities: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        raw_rows: list[dict[str, Any]] = []
        for row in entities.get("licenses", []):
            prepared = normalize_runtime_row(row, category="licenses")
            prepared["source_category"] = "licenses"
            prepared["price"] = float(prepared.get("price", 0.0) or 0.0)
            prepared["quantity"] = max(1, int(prepared.get("quantity", 1) or 1))
            raw_rows.append(prepared)

        for category in OPERATIONAL_COST_CATEGORIES:
            if category != "subscription_licenses":
                continue
            for row in entities.get(category, []):
                monthly_cost = float(row.get("monthly_cost", 0.0) or 0.0)
                if monthly_cost <= 0:
                    continue
                prepared = normalize_runtime_row(row, category=category)
                prepared["source_category"] = category
                prepared["price"] = monthly_cost * 12.0
                prepared["quantity"] = max(1, int(prepared.get("quantity", 1) or 1))
                raw_rows.append(prepared)

        if not raw_rows:
            return []

        prices = [float(row.get("price", 0.0) or 0.0) for row in raw_rows]
        min_price = min(prices)
        max_price = max(prices)
        span = max(max_price - min_price, 1.0)
        expanded: list[dict[str, Any]] = []
        for row in raw_rows:
            quality = (float(row.get("price", 0.0) or 0.0) - min_price) / span if span else 0.5
            template = self._software_template(row, quality=float(quality))
            for _ in range(max(1, int(row.get("quantity", 1) or 1))):
                expanded.append(deepcopy(template))
        expanded.sort(key=lambda item: (float(item.get("cost", 0.0)), str(item.get("source_name", ""))))
        return expanded

    def _device_template(self, row: Mapping[str, Any], *, role: str, quality: float) -> dict[str, Any]:
        profile = _ROLE_PROFILES[role]
        name = str(row.get("name", role)).strip() or role
        vendor = name.split()[0]
        template = {
            "role": role,
            "vendor": vendor,
            "source_name": name,
            "cpu_score": self._lerp(profile["cpu"], quality),
            "ram_score": self._lerp(profile["ram"], quality),
            "energy": self._lerp(profile["energy"], quality),
            "cost": round(float(row.get("price", 0.0) or 0.0), 2),
            "reliability": {
                "low": self._lerp(profile["rel_low"], quality, digits=3),
                "high": self._lerp(profile["rel_high"], quality, digits=3),
            },
            "lifespan": self._lerp(profile["lifespan"], quality),
            "client_seats": 0.0,
        }
        if role == "client":
            quantity = max(1.0, float(row.get("quantity", 1) or 1))
            if "client_seats" in row:
                seats_total = float(row.get("client_seats") or 0.0)
            elif row.get("component_type") == "workstation":
                seats_total = quantity
            else:
                seats_total = 0.0
            template["client_seats"] = seats_total / quantity
        return template

    def _software_template(self, row: Mapping[str, Any], *, quality: float) -> dict[str, Any]:
        profile = _ROLE_PROFILES["software"]
        name = str(row.get("name", "ПО")).strip() or "ПО"
        vendor = name.split()[0]
        functionality = self._lerp(profile["functionality"], quality)
        support = self._lerp(profile["support"], 1.0 - quality * 0.35)
        return {
            "role": "software",
            "vendor": vendor,
            "source_name": name,
            "cpu_score": functionality,
            "ram_score": support,
            "perf": functionality,
            "energy": 0.0,
            "cost": round(float(row.get("price", 0.0) or 0.0), 2),
            "reliability": {
                "low": self._lerp(profile["rel_low"], quality, digits=3),
                "high": self._lerp(profile["rel_high"], quality, digits=3),
            },
            "lifespan": self._lerp(profile["lifespan"], quality),
            "license_units": 1.0,
            "support_score": support,
            "source_category": row.get("source_category", "licenses"),
        }

    def _make_config(
        self,
        *,
        config_id: str,
        name: str,
        people: int,
        devices: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "id": config_id,
            "name": name,
            "devices": [deepcopy(device) for device in devices],
            "meta": {"people": people},
        }

    def _pick_units(self, units: list[dict[str, Any]], count: int, strategy: str) -> list[dict[str, Any]]:
        if not units or count <= 0:
            return []

        ordered = list(units)
        if strategy == "high":
            ordered = list(reversed(ordered))
        elif strategy == "mid":
            ordered = self._around_middle(units)
        elif strategy == "low_mid":
            ordered = units[: max(1, len(units) // 2 + 1)] + units[max(1, len(units) // 2 + 1) :]
        elif strategy == "mid_high":
            ordered = self._around_middle(units, prefer_high=True)

        selected: list[dict[str, Any]] = []
        for idx in range(count):
            selected.append(deepcopy(ordered[idx % len(ordered)]))
        return selected

    def _around_middle(self, units: list[dict[str, Any]], *, prefer_high: bool = False) -> list[dict[str, Any]]:
        if len(units) <= 2:
            return list(reversed(units)) if prefer_high else list(units)
        center = len(units) // 2
        offsets = [0]
        step = 1
        while len(offsets) < len(units):
            offsets.append(step)
            if len(offsets) < len(units):
                offsets.append(-step)
            step += 1
        ordered = []
        for offset in offsets:
            idx = min(max(center + offset, 0), len(units) - 1)
            item = units[idx]
            if item not in ordered:
                ordered.append(item)
        if prefer_high:
            high_tail = [item for item in reversed(units) if item not in ordered]
            ordered.extend(high_tail)
        else:
            low_tail = [item for item in units if item not in ordered]
            ordered.extend(low_tail)
        return ordered

    def _scaled_scores(self, values: list[float], *, reverse: bool = False) -> list[float]:
        if not values:
            return []
        low = min(values)
        high = max(values)
        if abs(high - low) < 1e-9:
            return [5.0 for _ in values]

        scaled = []
        for value in values:
            norm = (float(value) - low) / (high - low)
            if reverse:
                norm = 1.0 - norm
            scaled.append(round(2.0 + norm * 3.0, 2))
        return scaled

    def _client_seat_capacity(self, devices: Iterable[Mapping[str, Any]]) -> int:
        total = 0
        for device in devices:
            total += int(float(device.get("client_seats", 0.0) or 0.0))
        return total

    def _license_units(self, devices: Iterable[Mapping[str, Any]]) -> int:
        total = 0
        for device in devices:
            total += max(1, int(float(device.get("license_units", 1.0) or 1.0)))
        return total

    def _software_support_score(self, devices: Iterable[Mapping[str, Any]]) -> float:
        values = [float(device.get("support_score", 0.0) or 0.0) for device in devices]
        return float(median(values)) if values else 0.0

    def _lerp(self, bounds: tuple[float, float], position: float, digits: int = 2) -> float:
        lo, hi = bounds
        return round(lo + (hi - lo) * max(0.0, min(1.0, position)), digits)
