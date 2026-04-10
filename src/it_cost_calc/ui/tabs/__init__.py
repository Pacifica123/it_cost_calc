"""Публичные точки входа для вкладок интерфейса."""

from . import capex_tab as capex_tab
from . import configuration_selection_tab as configuration_selection_tab
from . import criteria_importance_tab as criteria_importance_tab
from . import energy_tab as energy_tab
from . import export_tab as export_tab
from . import infrastructure_tab as infrastructure_tab
from . import npv_tab as npv_tab
from . import opex_tab as opex_tab

# Совместимость со старыми импортами.
ahp_analysis_tab = configuration_selection_tab
ahp_tab = configuration_selection_tab
capital_costs_tab = capex_tab
capital_costs = capex_tab
electricity_costs_tab = energy_tab
electricity_costs = energy_tab
it_infrastructure = infrastructure_tab
npv_analysis = npv_tab
operational_costs_tab = opex_tab
operational_costs = opex_tab

__all__ = [
    "capex_tab",
    "configuration_selection_tab",
    "criteria_importance_tab",
    "energy_tab",
    "export_tab",
    "infrastructure_tab",
    "npv_tab",
    "opex_tab",
    "ahp_analysis_tab",
    "ahp_tab",
    "capital_costs_tab",
    "capital_costs",
    "electricity_costs_tab",
    "electricity_costs",
    "it_infrastructure",
    "npv_analysis",
    "operational_costs_tab",
    "operational_costs",
]
