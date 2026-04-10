"""Application service for electricity-cost calculation."""

from __future__ import annotations

from domain import ElectricityProfile, to_plain_data


class ElectricityCostService:
    def build_profiles(
        self,
        equipment_rows: list[ElectricityProfile | dict],
        hours_per_day: float,
        working_days: float,
        round_the_clock_names: set[str] | None = None,
    ) -> list[ElectricityProfile]:
        round_the_clock_names = round_the_clock_names or set()
        profiles: list[ElectricityProfile] = []

        for row in equipment_rows:
            payload = to_plain_data(row)
            name = str(payload.get("name", ""))
            is_round_the_clock = name in round_the_clock_names
            item_hours_per_day = 24.0 if is_round_the_clock else float(hours_per_day)
            item_working_days = 30.0 if is_round_the_clock else float(working_days)
            profiles.append(
                ElectricityProfile(
                    name=name,
                    quantity=float(payload.get("quantity", 0.0)),
                    max_power=float(payload.get("max_power", 0.0)),
                    hours_per_day=item_hours_per_day,
                    working_days=item_working_days,
                    round_the_clock=is_round_the_clock,
                )
            )

        return profiles

    def calculate(
        self,
        equipment_rows: list[ElectricityProfile | dict],
        hours_per_day: float,
        working_days: float,
        cost_per_kwh: float,
        round_the_clock_names: set[str] | None = None,
    ) -> dict:
        profiles = self.build_profiles(
            equipment_rows=equipment_rows,
            hours_per_day=hours_per_day,
            working_days=working_days,
            round_the_clock_names=round_the_clock_names,
        )

        total_cost = 0.0
        line_items: list[dict] = []
        for profile in profiles:
            cost = profile.calculate_cost(cost_per_kwh)
            total_cost += cost
            line_items.append(
                {
                    **to_plain_data(profile),
                    "energy_consumption": profile.energy_consumption,
                    "cost": cost,
                }
            )

        return {"total_cost": total_cost, "items": line_items}
