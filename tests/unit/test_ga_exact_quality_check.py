from domain.optimization.ga import run_ga_mvp
from domain.optimization.mech import Item


def _sum_prop(prop_name: str):
    def _inner(subset: list[Item]) -> float:
        return sum(float(item.properties.get(prop_name, 0.0)) for item in subset)

    return _inner


def _count_selected(subset: list[Item]) -> int:
    return len(subset)


def test_exact_quality_check_compares_ga_with_full_enumeration_on_small_input():
    items = [
        Item(name="A", value=10, cost=8),
        Item(name="B", value=7, cost=3),
        Item(name="C", value=4, cost=2),
    ]

    result = run_ga_mvp(
        {
            "items": items,
            "criteria": [
                {"name": "value", "func": _sum_prop("value"), "direction": "max"},
                {"name": "cost", "func": _sum_prop("cost"), "direction": "min"},
            ],
            "constraints": [
                {"name": "budget", "func": _sum_prop("cost"), "operator": "<=", "bound": 10},
                {"name": "min_selected", "func": _count_selected, "operator": ">=", "bound": 1},
            ],
            "weights": [0.75, 0.25],
            "pop_size": 6,
            "generations": 6,
            "mutation_rate": 0.1,
            "elite": 1,
            "seed": 11,
            "stagnation_limit": 10,
            "top_solutions_limit": 4,
            "quality_check_enabled": True,
            "quality_check_max_items": 10,
        }
    )

    quality_check = result["quality_check"]

    assert quality_check["method"] == "exhaustive_enumeration"
    assert quality_check["available"] is True
    assert quality_check["total_masks_checked"] == 8
    assert quality_check["feasible_count"] > 0
    assert quality_check["ga_best_matches_exact_best"] is True
    assert quality_check["ga_best_exact_rank"] == 1
    assert quality_check["exact_best_mask"] == result["best_mask"]
    assert quality_check["exact_best_score"] == result["best_agg"]
    assert quality_check["top_overlap_count"] == quality_check["top_solutions_limit"]
    assert quality_check["ga_top_matches_exact_top"] is True


def test_exact_quality_check_is_skipped_when_item_count_exceeds_limit():
    items = [Item(name=f"Item {index}", value=index + 1) for index in range(5)]

    result = run_ga_mvp(
        {
            "items": items,
            "criteria": [
                {"name": "value", "func": _sum_prop("value"), "direction": "max"},
            ],
            "constraints": [
                {"name": "min_selected", "func": _count_selected, "operator": ">=", "bound": 1},
            ],
            "weights": [1.0],
            "pop_size": 8,
            "generations": 4,
            "mutation_rate": 0.0,
            "elite": 1,
            "seed": 1,
            "stagnation_limit": 10,
            "quality_check_max_items": 4,
        }
    )

    quality_check = result["quality_check"]

    assert quality_check["status"] == "skipped"
    assert quality_check["available"] is False
    assert quality_check["reason"] == "too_many_items"
    assert quality_check["item_count"] == 5
