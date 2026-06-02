from application.services.analysis_scope_profile_service import AnalysisScopeProfileService
from application.services.genetic_optimization_service import GeneticOptimizationService


def test_scope_profiles_do_not_expose_category_coverage_as_soft_criterion():
    service = AnalysisScopeProfileService()

    for profile in service.profiles().values():
        assert "category_coverage" not in profile.criterion_ids()
        assert "category_coverage" not in profile.default_weights


def test_default_ga_fallback_keeps_categories_as_constraints_not_coverage_score():
    captured = {}

    def fake_runner(params):
        captured.update(params)
        return {
            "error": None,
            "best_items": [],
            "criteria": [],
            "constraints": [],
            "top_solutions": [],
            "criterion_names": [criterion["name"] for criterion in params["criteria"]],
            "constraints_metadata": [],
        }

    service = GeneticOptimizationService(
        {
            "server": [{"name": "Server", "quantity": 1, "price": 10}],
            "client": [{"name": "PC", "quantity": 1, "price": 5}],
        },
        ga_runner=fake_runner,
    )

    service.run(
        categories=("server", "client"),
        required_categories=("server",),
        excluded_categories=("network",),
    )

    criterion_names = [criterion["name"] for criterion in captured["criteria"]]
    constraint_names = [constraint["name"] for constraint in captured["constraints"]]

    assert criterion_names == ["selected_quantity", "capital_cost"]
    assert "category_coverage" not in criterion_names
    assert "required_category_server" in constraint_names
    assert "excluded_category_network" in constraint_names
