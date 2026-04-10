from it_cost_calc.application.services.equipment_service import EquipmentService
from it_cost_calc.domain import CapitalItem, OperationalExpense
from it_cost_calc.infrastructure.repositories.in_memory_entity_repository import (
    InMemoryEntityRepository,
)


def test_equipment_service_handles_capex_opex_and_energy_views():
    service = EquipmentService(InMemoryEntityRepository())

    service.add_capital_item("server", {"name": "srv-1", "quantity": 2, "price": 100.0})
    service.add_capital_item("client", {"name": "pc-1", "quantity": 5, "price": 50.0})
    service.add_operational_item("backup", {"name": "cloud", "monthly_cost": 10.0})

    server_item = service.list_capital_items("server")[0]
    backup_item = service.list_operational_items("backup")[0]

    assert isinstance(server_item, CapitalItem)
    assert isinstance(backup_item, OperationalExpense)
    assert server_item.name == "srv-1"
    assert backup_item.monthly_cost == 10.0
    assert {row.name for row in service.list_energy_relevant_items()} == {"srv-1", "pc-1"}
    assert service.list_round_the_clock_equipment_names() == {"srv-1"}
