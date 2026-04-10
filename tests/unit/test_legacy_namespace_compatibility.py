from bootstrap import load_demo_data as top_level_load_demo_data
from domain import CapitalItem as top_level_capital_item
from application.services.equipment_service import EquipmentService as top_level_equipment_service

from it_cost_calc.bootstrap import load_demo_data as legacy_load_demo_data
from it_cost_calc.domain import CapitalItem as legacy_capital_item
from it_cost_calc.application.services.equipment_service import EquipmentService as legacy_equipment_service


def test_legacy_namespace_proxies_to_top_level_modules():
    assert legacy_load_demo_data is top_level_load_demo_data
    assert legacy_capital_item is top_level_capital_item
    assert legacy_equipment_service is top_level_equipment_service
