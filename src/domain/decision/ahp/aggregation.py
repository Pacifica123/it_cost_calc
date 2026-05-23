from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .models import FuzzyInterval


def load_configurations_from_json(text: str) -> List[Dict[str, Any]]:
    data = json.loads(text)
    if isinstance(data, dict) and "configurations" in data:
        cfgs = data["configurations"]
    elif isinstance(data, list):
        cfgs = data
    else:
        raise ValueError(
            'JSON должен содержать либо список конфигураций, либо объект с ключом "configurations"'
        )
    return cfgs


def aggregate_configuration(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    На вход — конфигурация в формате:
    {
      "id": "A",
      "devices": [ {"role":"client","vendor":"AMD", "cpu_score":2.0, "energy":50, "cost":300, "reliability": {"low":0.6,"high":0.8} }, ... ],
      "meta": {"people":3}
    }

    Возвращает агрегированные характеристики конфигурации:
    - total_cost
    - total_energy
    - avg_reliability (дефаззификация центроидом)
    - total_performance (сумма cpu_score + ram_score... если нет — 0)
    - counts per role (clients, servers, routers)
    - people (из meta или 0)
    """
    devs = cfg.get("devices", [])
    if not devs and isinstance(cfg.get("components"), list):
        devs = cfg.get("components", [])
    total_cost = 0.0
    total_energy = 0.0
    perf_sum = 0.0
    rel_vals = []
    lifespan_vals = []
    counts = {}
    for d in devs:
        role = d.get("role") or d.get("source_category") or d.get("category") or d.get("component_type") or "unknown"
        if role == "client" and "client_seats" in d:
            counts[role] = counts.get(role, 0.0) + float(d.get("client_seats", 0.0) or 0.0)
        elif role == "software" and "license_units" in d:
            counts[role] = counts.get(role, 0.0) + float(d.get("license_units", 1.0) or 0.0)
        else:
            counts[role] = counts.get(role, 0.0) + 1.0
        quantity = float(d.get("quantity", 1.0) or 1.0)
        unit_price = float(d.get("price", d.get("unit_price", 0.0)) or 0.0)
        total_cost += float(d.get("cost", d.get("total_cost", quantity * unit_price)) or 0.0)
        total_energy += float(d.get("energy", d.get("max_power", 0.0)) or 0.0)
        # производительность — суммируем поля, если они есть
        # допустим поля cpu_score и ram_score и custom perf
        perf_sum += (
            float(d.get("cpu_score", 0.0))
            + float(d.get("ram_score", 0.0))
            + float(d.get("perf", 0.0))
        )
        r = d.get("reliability", None)
        rel_centroid = None
        if isinstance(r, dict) and "low" in r and "high" in r:
            rel_centroid = FuzzyInterval(float(r["low"]), float(r["high"])).centroid()
            rel_vals.append(rel_centroid)
        else:
            if r is None:
                pass
            else:
                try:
                    rel_centroid = float(r)
                    rel_vals.append(rel_centroid)
                except Exception:
                    pass

        lifespan = d.get("lifespan", None)
        if lifespan is not None:
            try:
                lifespan_vals.append(float(lifespan))
            except Exception:
                pass
    avg_reliability = float(np.mean(rel_vals)) if len(rel_vals) > 0 else 0.0
    avg_lifespan = float(np.mean(lifespan_vals)) if len(lifespan_vals) > 0 else 0.0
    totals = cfg.get("totals", {}) if isinstance(cfg.get("totals"), dict) else {}
    metrics = cfg.get("metrics", {}) if isinstance(cfg.get("metrics"), dict) else {}
    candidate_meta = cfg.get("metadata", {}) if isinstance(cfg.get("metadata"), dict) else {}
    legacy_meta = candidate_meta.get("legacy_meta", {}) if isinstance(candidate_meta.get("legacy_meta"), dict) else {}
    meta = cfg.get("meta", {}) if isinstance(cfg.get("meta"), dict) else legacy_meta
    res = {
        "id": cfg.get("id", None),
        "total_cost": float(totals.get("total_cost", totals.get("capital_cost", total_cost)) or 0.0),
        "total_energy": float(totals.get("total_energy", totals.get("energy", total_energy)) or 0.0),
        "total_performance": float(metrics.get("total_performance", perf_sum) or 0.0),
        "avg_reliability": float(metrics.get("avg_reliability", avg_reliability) or 0.0),
        "counts": dict(metrics.get("counts", counts) or {}),
        "people": int(metrics.get("people", meta.get("people", cfg.get("people", 0))) or 0),
        "lifespan": float(metrics.get("lifespan", avg_lifespan) or 0.0),
    }
    return res


def filter_hard_constraints(
    aggregates: Sequence[Dict[str, Any]], constraints: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    constraints может содержать:
      - max_budget
      - max_energy
      - people_match_tolerance: допустимое число лишних клиентских ПК в процентах (например 0.2 = 20%)

    Возвращает (passed, removed)
    """
    passed = []
    removed = []
    for a in aggregates:
        fail_reasons = []
        if (
            "max_budget" in constraints
            and a["total_cost"] > float(constraints["max_budget"]) + 1e-9
        ):
            fail_reasons.append("budget")
        if (
            "max_energy" in constraints
            and a["total_energy"] > float(constraints["max_energy"]) + 1e-9
        ):
            fail_reasons.append("energy")
        # people vs client seats for technical equipment.
        # client_seats allows rows like MFP/printer to stay in the client category
        # without pretending that they replace a human workplace.
        people = int(a.get("people", 0))
        capacity_role = str(constraints.get("capacity_role", "client"))
        capacity = float(a.get("counts", {}).get(capacity_role, 0.0))
        tol = float(constraints.get("people_match_tolerance", 0.2))
        if people > 0 and capacity_role == "client":
            if capacity < people:
                fail_reasons.append("not_enough_clients")
            else:
                excess = capacity - people
                if excess > math.ceil(tol * people):
                    fail_reasons.append("too_many_clients")
        # if people == 0 or the scope is software-only, user/workplace matching is skipped.
        if len(fail_reasons) == 0:
            passed.append(a)
        else:
            removed.append({"aggregate": a, "reasons": fail_reasons})
    return passed, removed
