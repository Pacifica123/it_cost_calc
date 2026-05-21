from domain.optimization.ga import run_ga_mvp
from domain.optimization.mech import Item


def _sum_prop(prop_name: str):
    def _inner(subset: list[Item]) -> float:
        return sum(float(item.properties.get(prop_name, 0.0)) for item in subset)

    return _inner


def _count_selected(subset: list[Item]) -> int:
    return len(subset)


def test_top_solutions_are_built_from_unique_feasible_archive_not_only_final_population():
    """GA must not lose good candidates that were seen before the final population.

    With pop_size=1 the old implementation could return at most the global best
    and the last living mask. The archive-based implementation preserves all
    unique feasible singleton solutions generated during heuristic initialization
    and ranks top_solutions from that archive.
    """

    items = [
        Item(name="A", value=10),
        Item(name="B", value=9),
        Item(name="C", value=8),
    ]

    result = run_ga_mvp(
        {
            "items": items,
            "criteria": [
                {"name": "value", "func": _sum_prop("value"), "direction": "max"}
            ],
            "constraints": [
                {"name": "min_selected", "func": _count_selected, "operator": ">=", "bound": 1},
                {"name": "single_item", "func": _count_selected, "operator": "<=", "bound": 1},
            ],
            "weights": [1.0],
            "pop_size": 1,
            "generations": 4,
            "mutation_rate": 1.0,
            "elite": 0,
            "seed": 3,
            "stagnation_limit": 10,
            "top_solutions_limit": 3,
            "max_random_attempts": 50,
        }
    )

    assert result["error"] is None
    assert result["top_solutions_source"] == "unique_feasible_archive"
    assert result["unique_feasible_solutions_seen"] >= 3
    assert [solution["items"][0]["name"] for solution in result["top_solutions"]] == [
        "A",
        "B",
        "C",
    ]
    assert [solution["rank"] for solution in result["top_solutions"]] == [1, 2, 3]
