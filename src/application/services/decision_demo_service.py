from __future__ import annotations

from copy import deepcopy
from statistics import median
from typing import Any, Iterable, Mapping

from domain.decision.ahp.aggregation import aggregate_configuration

CRITERIA_IMPORTANCE_CRITERIA = [
    {"id": "1", "name": "Надежность и отказоустойчивость"},
    {"id": "2", "name": "Производительность конфигурации"},
    {"id": "3", "name": "Стоимость владения"},
    {"id": "4", "name": "Энергоэффективность"},
    {"id": "5", "name": "Потенциал развития и срок службы"},
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
}


class DecisionDemoDataService:
    """Builds consistent demo data for AHP and criteria-importance tabs from runtime entities."""

    def build(self, entities: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
        configurations = self.build_ahp_configurations(entities)
        constraints = self.build_recommended_constraints(configurations)
        criteria_case = self.build_criteria_importance_case(configurations)
        return {
            "configurations": configurations,
            "constraints": constraints,
            "criteria_case": criteria_case,
        }

    def build_ahp_configurations(
        self, entities: Mapping[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        servers = self._expand_category(entities.get("server", []), role="server")
        clients = self._expand_category(entities.get("client", []), role="client")
        networks = self._expand_category(entities.get("network", []), role="network")

        if not servers or not clients:
            raise ValueError(
                "Для подготовки демонстрационных конфигураций нужны серверы и клиентские устройства"
            )

        client_target = max(2, len(clients))
        configs = [
            self._make_config(
                config_id="cfg_budget",
                name="Экономичная конфигурация",
                people=max(2, client_target - 1),
                servers=self._pick_units(servers, 1, "low"),
                clients=self._pick_units(clients, max(2, client_target - 1), "low"),
                networks=self._pick_units(networks, 1, "low"),
            ),
            self._make_config(
                config_id="cfg_compact",
                name="Компактная конфигурация",
                people=client_target,
                servers=self._pick_units(servers, 1, "mid"),
                clients=self._pick_units(clients, client_target, "low_mid"),
                networks=self._pick_units(networks, 1, "mid"),
            ),
            self._make_config(
                config_id="cfg_balanced",
                name="Сбалансированная конфигурация",
                people=client_target,
                servers=self._pick_units(servers, 1, "mid"),
                clients=self._pick_units(clients, client_target, "mid"),
                networks=self._pick_units(networks, min(2, max(len(networks), 1)), "mid"),
            ),
            self._make_config(
                config_id="cfg_growth",
                name="Конфигурация с запасом роста",
                people=client_target,
                servers=self._pick_units(servers, min(2, len(servers)), "high"),
                clients=self._pick_units(clients, client_target, "high"),
                networks=self._pick_units(networks, min(2, max(len(networks), 1)), "high"),
            ),
            self._make_config(
                config_id="cfg_reserve",
                name="Резервоустойчивая конфигурация",
                people=client_target,
                servers=self._pick_units(servers, max(1, min(2, len(servers))), "high"),
                clients=self._pick_units(clients, client_target + 1, "mid_high"),
                networks=self._pick_units(networks, max(1, min(2, len(networks))), "high"),
            ),
        ]
        return configs

    def build_recommended_constraints(
        self, configurations: Iterable[dict[str, Any]]
    ) -> dict[str, float]:
        aggregates = [aggregate_configuration(config) for config in configurations]
        costs = sorted(float(item["total_cost"]) for item in aggregates)
        energies = sorted(float(item["total_energy"]) for item in aggregates)
        if not costs or not energies:
            raise ValueError("Не удалось вычислить ограничения для пустого набора конфигураций")

        pivot_index = min(len(costs) - 1, 3)
        max_budget = round(costs[pivot_index] * 1.05, 2)
        max_energy = round(energies[pivot_index] * 1.08, 2)
        return {
            "max_budget": max_budget,
            "max_energy": max_energy,
            "people_match_tolerance": 0.25,
        }

    def build_criteria_importance_case(
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

        scores: dict[str, dict[str, float]] = {}
        for idx, alt in enumerate(alternatives):
            scores[alt["id"]] = {
                criterion_id: scores_values[idx]
                for criterion_id, scores_values in scores_map.items()
            }

        return {
            "name": "Выбор конфигурации ИТ-инфраструктуры",
            "criteria": deepcopy(CRITERIA_IMPORTANCE_CRITERIA),
            "alternatives": alternatives,
            "scores": scores,
            "relations": deepcopy(CRITERIA_IMPORTANCE_RELATIONS),
        }

    def _expand_category(
        self, rows: Iterable[Mapping[str, Any]], *, role: str
    ) -> list[dict[str, Any]]:
        prepared_rows = [row for row in rows if float(row.get("price", 0.0) or 0.0) > 0]
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

    def _device_template(self, row: Mapping[str, Any], *, role: str, quality: float) -> dict[str, Any]:
        profile = _ROLE_PROFILES[role]
        name = str(row.get("name", role)).strip() or role
        vendor = name.split()[0]
        return {
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
        }

    def _make_config(
        self,
        *,
        config_id: str,
        name: str,
        people: int,
        servers: list[dict[str, Any]],
        clients: list[dict[str, Any]],
        networks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        devices = [
            *(deepcopy(device) for device in servers),
            *(deepcopy(device) for device in clients),
            *(deepcopy(device) for device in networks),
        ]
        return {
            "id": config_id,
            "name": name,
            "devices": devices,
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

    def _lerp(self, bounds: tuple[float, float], position: float, digits: int = 2) -> float:
        lo, hi = bounds
        return round(lo + (hi - lo) * max(0.0, min(1.0, position)), digits)
