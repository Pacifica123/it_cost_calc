from it_cost_calc.application.crud_compat import CRUD as CrudCompat
from it_cost_calc.application.legacy_crud import CRUD as LegacyCrud
from it_cost_calc.domain.decision.ahp.configuration_selector import (
    run_ahp_pipeline as canonical_ahp_pipeline,
)
from it_cost_calc.domain.decision.ahp.ahp_config_selector import (
    run_ahp_pipeline as legacy_ahp_pipeline,
)
from it_cost_calc.domain.decision.criteria_importance.analysis import (
    run_importance_pipeline as canonical_importance_pipeline,
)
from it_cost_calc.domain.decision.criteria_importance.importance_method import (
    run_importance_pipeline as legacy_importance_pipeline,
)
from it_cost_calc.ui.tabs.capex_tab import CapexTab, CapitalCostsTab
from it_cost_calc.ui.tabs.opex_tab import OpexTab, OperationalCostsTab
from it_cost_calc.ui.tabs.energy_tab import EnergyTab, ElectricityCostsTab
from it_cost_calc.ui.tabs.configuration_selection_tab import (
    ConfigurationSelectionTab,
    AHPAnalysisTab,
)


def test_canonical_and_legacy_imports_point_to_same_objects():
    assert CrudCompat is LegacyCrud
    assert canonical_ahp_pipeline is legacy_ahp_pipeline
    assert canonical_importance_pipeline is legacy_importance_pipeline
    assert CapexTab is CapitalCostsTab
    assert OpexTab is OperationalCostsTab
    assert EnergyTab is ElectricityCostsTab
    assert ConfigurationSelectionTab is AHPAnalysisTab
