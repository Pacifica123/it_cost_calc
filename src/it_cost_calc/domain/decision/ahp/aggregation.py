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
    total_cost = 0.0
    total_energy = 0.0
    perf_sum = 0.0
    rel_vals = []
    counts = {}
    for d in devs:
        role = d.get("role", "unknown")
        counts[role] = counts.get(role, 0) + 1
        total_cost += float(d.get("cost", 0.0))
        total_energy += float(d.get("energy", 0.0))
        # производительность — суммируем поля, если они есть
        # допустим поля cpu_score и ram_score и custom perf
        perf_sum += (
            float(d.get("cpu_score", 0.0))
            + float(d.get("ram_score", 0.0))
            + float(d.get("perf", 0.0))
        )
        r = d.get("reliability", None)
        if isinstance(r, dict) and "low" in r and "high" in r:
            rel_vals.append(FuzzyInterval(float(r["low"]), float(r["high"])).centroid())
        else:
            if r is None:
                pass
            else:
                try:
                    rel_vals.append(float(r))
                except Exception:
                    pass
    avg_reliability = float(np.mean(rel_vals)) if len(rel_vals) > 0 else 0.0
    res = {
        "id": cfg.get("id", None),
        "total_cost": total_cost,
        "total_energy": total_energy,
        "total_performance": perf_sum,
        "avg_reliability": avg_reliability,
        "counts": counts,
        "people": int(cfg.get("meta", {}).get("people", cfg.get("people", 0))),
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
        # people vs client pcs
        people = int(a.get("people", 0))
        clients = int(a.get("counts", {}).get("client", 0))
        tol = float(constraints.get("people_match_tolerance", 0.2))
        # требуем clients >= people; и клиенты сверх людей не должны превышать tol*people
        if people > 0:
            if clients < people:
                fail_reasons.append("not_enough_clients")
            else:
                excess = clients - people
                if excess > math.ceil(tol * people):
                    fail_reasons.append("too_many_clients")
        # если люди == 0 — допускаем любые клиенты
        if len(fail_reasons) == 0:
            passed.append(a)
        else:
            removed.append({"aggregate": a, "reasons": fail_reasons})
    return passed, removed
