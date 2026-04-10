from domain import (
    AnalysisResult,
    CapitalItem,
    Criterion,
    ElectricityProfile,
    EquipmentCategory,
    ExpenseType,
    OperationalExpense,
    Relation,
    RelationOperator,
    to_plain_data,
)


def test_domain_models_serialize_cleanly_to_plain_data():
    capital = CapitalItem(name="srv", quantity=2, price=100.0, category=EquipmentCategory.SERVER)
    operational = OperationalExpense(name="backup", monthly_cost=15.0)
    electricity = ElectricityProfile(
        name="srv",
        quantity=2,
        max_power=150.0,
        hours_per_day=24.0,
        working_days=30.0,
        round_the_clock=True,
    )
    relation = Relation(left="1", operator=RelationOperator.GREATER, factor=2.0, right="2")
    result = AnalysisResult(
        case_name="demo",
        ranking=[{"id": "alt-1", "name": "Alt 1", "score": 0.9}],
        weights={"alt-1": 0.9},
        winner_id="alt-1",
        winner_name="Alt 1",
        explanation="ok",
        metadata={"criterion": Criterion(id="1", name="Стоимость")},
    )

    payload = to_plain_data(
        {
            "capital": capital,
            "operational": operational,
            "electricity": electricity,
            "relation": relation,
            "result": result,
        }
    )

    assert payload["capital"]["category"] == EquipmentCategory.SERVER.value
    assert payload["capital"]["expense_type"] == ExpenseType.CAPITAL.value
    assert payload["operational"]["expense_type"] == ExpenseType.OPERATIONAL.value
    assert payload["relation"]["operator"] == RelationOperator.GREATER.value
    assert payload["result"]["metadata"]["criterion"]["name"] == "Стоимость"
