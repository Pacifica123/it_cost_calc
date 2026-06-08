from domain.decision.ahp.configuration_selector import (
    run_ahp_pipeline as canonical_ahp_pipeline,
)
from domain.decision.ahp.ahp_config_selector import (
    run_ahp_pipeline as legacy_ahp_pipeline,
)
from domain.decision.criteria_importance.analysis import (
    run_importance_pipeline as canonical_importance_pipeline,
)
from domain.decision.criteria_importance.importance_method import (
    run_importance_pipeline as legacy_importance_pipeline,
)


def test_canonical_and_legacy_domain_imports_point_to_same_objects():
    assert canonical_ahp_pipeline is legacy_ahp_pipeline
    assert canonical_importance_pipeline is legacy_importance_pipeline
