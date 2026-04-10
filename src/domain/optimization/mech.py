"""Compatibility wrapper for the refactored optimization mechanic."""

from __future__ import annotations

from typing import Any, Dict

from .mech_models import FuzzyTerm, FuzzyValue, Item
from .mech_solver import KnapsackMechanic
from .mech_utils import to_json_serializable


def run(params: Dict[str, Any]) -> Dict[str, Any]:
    solver = KnapsackMechanic(
        items=params["items"],
        max_constraint=params["max_constraint"],
        constraint_func=params["constraint_func"],
        criteria_funcs=params["criteria_funcs"],
        optimize_directions=params["optimize_directions"],
        pairwise_matrix=params["pairwise_matrix"],
        exclude_empty=params.get("exclude_empty", True),
        dominance_eps=params.get("dominance_eps", 1e-9),
    )
    return solver.solve()


__all__ = [
    "Item",
    "FuzzyValue",
    "FuzzyTerm",
    "KnapsackMechanic",
    "to_json_serializable",
    "run",
]
