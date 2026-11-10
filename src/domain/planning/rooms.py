"""Cost of the premises part: structured cabling, UPS, climate and lighting.

Simplified engineering estimates for rooms described by size and equipment counts
(no floor plan): enough to put the infrastructure of the rooms into the budget.

* Cabling: one cable from every device to the rack; run = half perimeter + 3 m drop and slack.
* Climate: Q = V·q + N·100 W + ΣP equipment, required = Q·1,15, split units of 3,5 kW.
* Lighting: N = E·S·Kз / (Φ·η), η from the room index i = S / (h·(a + b)).
* UPS: the smallest standard model that carries the load with a 30 % reserve and the runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

WALL_HEIGHT_M = 2.7
WORK_SURFACE_M = 0.8
DEVICE_POWER_W = {"workstation": 250, "printer": 350, "server": 400, "network": 40, "wifi": 10}
PERSON_HEAT_W = 100
BUILDING_HEAT_W_PER_M3 = {"no_windows": 30, "windows": 35}
COOLING_RESERVE = 1.15
AIRCON_COOLING_W = 3500
AIRCON_POWER_W = 1000
ILLUMINANCE_LX = {"pc": 400, "office": 300, "technical": 200}
MAINTENANCE_FACTOR = 1.4
LAN_BOX_M = 305
PATCH_PANEL_PORTS = 24
CHANNEL_PIECE_M = 2


@dataclass(frozen=True)
class RoomPrices:
    lan_cable_per_m: float = 35
    power_cable_per_m: float = 60
    lan_socket: float = 450
    power_socket: float = 350
    connector: float = 25
    patch_cord: float = 150
    channel_piece: float = 300
    patch_panel: float = 4500
    aircon: float = 45000
    aircon_install: float = 15000
    luminaire: float = 2500
    luminaire_flux_lm: float = 3600
    luminaire_power_w: float = 36
    install_per_point: float = 600
    tariff_per_kwh: float = 8.0


@dataclass(frozen=True)
class Room:
    name: str
    width_m: float
    length_m: float
    workstations: int = 0
    printers: int = 0
    servers: int = 0
    network_devices: int = 0
    wifi_points: int = 0
    has_windows: bool = True
    server_room: bool = False

    @property
    def area(self) -> float:
        return max(0.0, self.width_m) * max(0.0, self.length_m)


@dataclass(frozen=True)
class CostLine:
    group: str
    name: str
    unit: str
    quantity: float
    price: float

    @property
    def total(self) -> float:
        return round(self.quantity * self.price, 2)


@dataclass(frozen=True)
class RoomEstimate:
    room: Room
    lan_cables: int
    lan_meters: float
    heat_w: float
    required_cooling_w: float
    aircons: int
    lux: int
    luminaires: int
    achieved_lux: int
    lighting_power_w: float
    monthly_kwh: float
    lines: tuple[CostLine, ...] = field(default_factory=tuple)

    @property
    def total(self) -> float:
        return round(sum(line.total for line in self.lines), 2)


def utilization_factor(room_index: float) -> float:
    for limit, value in ((0.6, 0.3), (0.8, 0.36), (1.0, 0.42), (1.25, 0.47), (1.5, 0.51), (2.0, 0.56), (2.5, 0.6), (3.0, 0.63)):
        if room_index < limit:
            return value
    return 0.66


def estimate_room(room: Room, prices: RoomPrices | None = None) -> RoomEstimate:
    prices = prices or RoomPrices()
    width, length = max(0.5, room.width_m), max(0.5, room.length_m)
    area = width * length
    lines: list[CostLine] = []

    # Cabling
    lan_cables = room.workstations + room.printers + room.servers + room.wifi_points
    run = (width + length) / 2 + 3
    lan_meters = math.ceil(lan_cables * run)
    lan_sockets = room.workstations + room.printers
    power_sockets = room.workstations * 2 + room.printers + room.servers * 2 + (2 if room.network_devices else 0)
    power_meters = math.ceil(2 * (width + length) * 1.1 + power_sockets) if power_sockets else 0
    channel_meters = min(2 * (width + length), max(lan_meters, power_meters))
    if lan_cables:
        boxes = math.ceil(lan_meters / LAN_BOX_M)
        lines += [
            CostLine("СКС", f"Кабель U/UTP кат. 6 ({boxes} бухт по {LAN_BOX_M} м)", "м", lan_meters, prices.lan_cable_per_m),
            CostLine("СКС", "Коннектор RJ-45 кат. 6 (+10 %)", "шт.", math.ceil(lan_cables * 2 * 1.1), prices.connector),
            CostLine("СКС", "Патч-корд кат. 6", "шт.", lan_cables * 2, prices.patch_cord),
            CostLine("СКС", f"Патч-панель {PATCH_PANEL_PORTS} порта", "шт.", math.ceil(lan_cables / PATCH_PANEL_PORTS), prices.patch_panel),
            CostLine("СКС", "Монтаж линии", "шт.", lan_cables, prices.install_per_point),
        ]
    if lan_sockets:
        lines.append(CostLine("СКС", "Розетка RJ-45 кат. 6", "шт.", lan_sockets, prices.lan_socket))
    if power_sockets:
        lines += [
            CostLine("Электрика", "Кабель ВВГнг-LS 3×2,5", "м", power_meters, prices.power_cable_per_m),
            CostLine("Электрика", "Розетка 220 В с заземлением", "шт.", power_sockets, prices.power_socket),
        ]
    if channel_meters:
        lines.append(CostLine("СКС", "Кабель-канал двухсекционный, 2 м", "шт.", math.ceil(channel_meters / CHANNEL_PIECE_M), prices.channel_piece))

    # Climate
    volume = area * WALL_HEIGHT_M
    building = volume * BUILDING_HEAT_W_PER_M3["windows" if room.has_windows else "no_windows"]
    people = (room.workstations if not room.server_room else 0) * PERSON_HEAT_W
    equipment = (
        room.workstations * DEVICE_POWER_W["workstation"]
        + room.printers * DEVICE_POWER_W["printer"]
        + room.servers * DEVICE_POWER_W["server"]
        + room.network_devices * DEVICE_POWER_W["network"]
        + room.wifi_points * DEVICE_POWER_W["wifi"]
    )
    heat = building + people + equipment
    required = heat * COOLING_RESERVE
    aircons = max(1, math.ceil(required / AIRCON_COOLING_W)) if (room.server_room or room.servers or room.workstations) else 0
    if aircons:
        lines.append(CostLine("Климат", "Сплит-система 3,5 кВт с монтажом", "шт.", aircons, prices.aircon + prices.aircon_install))

    # Lighting
    has_pcs = room.workstations > 0 and not room.server_room
    technical = room.server_room or (not room.workstations and room.servers > 0)
    lux = ILLUMINANCE_LX["pc"] if has_pcs else ILLUMINANCE_LX["technical"] if technical else ILLUMINANCE_LX["office"]
    height = WALL_HEIGHT_M - WORK_SURFACE_M
    index = area / (height * (width + length))
    eta = utilization_factor(index)
    luminaires = max(1, math.ceil(lux * area * MAINTENANCE_FACTOR / (prices.luminaire_flux_lm * eta)))
    achieved = round(luminaires * prices.luminaire_flux_lm * eta / (area * MAINTENANCE_FACTOR))
    lighting_w = luminaires * prices.luminaire_power_w
    lines.append(CostLine("Освещение", "Светодиодная панель 595×595 с монтажом", "шт.", luminaires, prices.luminaire))

    # Energy of light and air conditioning per month
    light_hours = 2 * 22 if technical else 8 * 22
    aircon_hours = 24 * 30 if technical else 8 * 22
    monthly_kwh = round((lighting_w * light_hours + aircons * AIRCON_POWER_W * aircon_hours * 0.5) / 1000, 1)

    return RoomEstimate(
        room=room,
        lan_cables=lan_cables,
        lan_meters=lan_meters,
        heat_w=round(heat),
        required_cooling_w=round(required),
        aircons=aircons,
        lux=lux,
        luminaires=luminaires,
        achieved_lux=achieved,
        lighting_power_w=lighting_w,
        monthly_kwh=monthly_kwh,
        lines=tuple(lines),
    )


# --- UPS -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class UpsModel:
    va: int
    watts: int
    batteries: int
    battery_ah: int
    price: float
    pack_price: float


UPS_MODELS = (
    UpsModel(650, 390, 1, 7, 6000, 0),
    UpsModel(1000, 600, 2, 7, 11000, 0),
    UpsModel(1500, 900, 2, 9, 15000, 0),
    UpsModel(2000, 1800, 4, 9, 32000, 14000),
    UpsModel(3000, 2700, 6, 9, 48000, 18000),
    UpsModel(6000, 5400, 16, 9, 120000, 42000),
    UpsModel(10000, 9000, 20, 9, 190000, 52000),
)
UPS_LOAD_RESERVE = 1.3
UPS_BATTERY_K = 0.7
BATTERY_VOLTAGE = 12


@dataclass(frozen=True)
class UpsChoice:
    load_w: float
    model: UpsModel | None
    packs: int
    runtime_min: float
    price: float


def ups_runtime_minutes(model: UpsModel, packs: int, load_w: float) -> float:
    if load_w <= 0:
        return 0.0
    energy_wh = model.batteries * (1 + packs) * BATTERY_VOLTAGE * model.battery_ah
    return round(energy_wh * UPS_BATTERY_K / load_w * 60, 1)


def choose_ups(load_w: float, target_min: float) -> UpsChoice:
    """Cheapest standard set (model + external packs ≤ 6) that carries load × 1,3 for the target runtime."""
    if load_w <= 0:
        return UpsChoice(0, None, 0, 0, 0)
    best: tuple[float, UpsModel, int] | None = None
    for model in UPS_MODELS:
        if model.watts < load_w * UPS_LOAD_RESERVE:
            continue
        for packs in range(0, 7 if model.pack_price else 1):
            if ups_runtime_minutes(model, packs, load_w) >= target_min:
                price = model.price + packs * model.pack_price
                if best is None or price < best[0]:
                    best = (price, model, packs)
                break
    if best is None:
        return UpsChoice(load_w, None, 0, 0, 0)
    price, model, packs = best
    return UpsChoice(load_w, model, packs, ups_runtime_minutes(model, packs, load_w), price)


def critical_load_w(rooms: Iterable[Room], *, include_workstations: bool = False) -> float:
    total = 0.0
    for room in rooms:
        total += room.servers * DEVICE_POWER_W["server"] + room.network_devices * DEVICE_POWER_W["network"] + room.wifi_points * DEVICE_POWER_W["wifi"]
        if include_workstations:
            total += room.workstations * DEVICE_POWER_W["workstation"]
    return total


@dataclass(frozen=True)
class PremisesEstimate:
    rooms: tuple[RoomEstimate, ...]
    ups: UpsChoice
    total: float
    by_group: dict[str, float]
    monthly_kwh: float
    monthly_energy_cost: float


def estimate_premises(rooms: Sequence[Room], prices: RoomPrices | None = None, *, ups_minutes: float = 15, ups_for_workstations: bool = False) -> PremisesEstimate:
    prices = prices or RoomPrices()
    estimates = tuple(estimate_room(room, prices) for room in rooms)
    ups = choose_ups(critical_load_w(rooms, include_workstations=ups_for_workstations), ups_minutes)
    by_group: dict[str, float] = {}
    for estimate in estimates:
        for line in estimate.lines:
            by_group[line.group] = round(by_group.get(line.group, 0.0) + line.total, 2)
    if ups.price:
        by_group["ИБП"] = ups.price
    total = round(sum(by_group.values()), 2)
    kwh = round(sum(estimate.monthly_kwh for estimate in estimates), 1)
    return PremisesEstimate(estimates, ups, total, by_group, kwh, round(kwh * prices.tariff_per_kwh, 2))


def rooms_from_counts(workstations: int, servers: int, network_devices: int, printers: int = 0) -> list[Room]:
    """Default rooms for the selected configuration: an office and, with servers, a server room."""
    rooms: list[Room] = []
    if workstations or printers:
        area = max(12.0, workstations * 4.5)
        width = round(max(3.0, math.sqrt(area / 1.5)), 1)
        length = round(area / width, 1)
        wifi = 1 + workstations // 15 if workstations else 0
        rooms.append(Room("Офис", width, length, workstations, printers, 0, 0 if servers else max(1, network_devices), wifi, True, False))
    if servers:
        rooms.append(Room("Серверная", 3.0, 3.0, 0, 0, servers, max(1, network_devices), 0, False, True))
    return rooms
