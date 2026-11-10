"""Presenter for the «Помещения» tab: cost of cabling, UPS, climate and lighting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from domain.finance.financing import count_devices
from domain.planning.rooms import PremisesEstimate, Room, RoomPrices, estimate_premises, rooms_from_counts
from ui_qt.presenters.app_presenter import QtAppPresenter
from ui_qt.presenters.planning_presenter import money
from ui_qt.presenters.session_extras import load_extra, save_extra

SESSION_KEY = "rooms_settings"

ROOM_COLUMNS: tuple[tuple[str, str], ...] = (
    ("name", "Помещение"),
    ("width_m", "Ширина, м"),
    ("length_m", "Длина, м"),
    ("workstations", "ПК"),
    ("printers", "МФУ"),
    ("servers", "Серверы"),
    ("network_devices", "Сеть"),
    ("wifi_points", "Wi-Fi"),
    ("has_windows", "Окна"),
    ("server_room", "Серверная"),
)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "да", "yes", "true", "+", "y", "д")


def room_from_mapping(payload: Mapping[str, Any]) -> Room | None:
    try:
        width = max(0.5, min(200.0, float(str(payload.get("width_m", 0)).replace(",", "."))))
        length = max(0.5, min(200.0, float(str(payload.get("length_m", 0)).replace(",", "."))))
    except (TypeError, ValueError):
        return None

    def count(key: str) -> int:
        try:
            return max(0, min(10000, int(float(str(payload.get(key, 0) or 0).replace(",", ".")))))
        except (TypeError, ValueError):
            return 0

    return Room(
        name=str(payload.get("name") or "Помещение").strip() or "Помещение",
        width_m=width,
        length_m=length,
        workstations=count("workstations"),
        printers=count("printers"),
        servers=count("servers"),
        network_devices=count("network_devices"),
        wifi_points=count("wifi_points"),
        has_windows=_bool(payload.get("has_windows", True)),
        server_room=_bool(payload.get("server_room", False)),
    )


@dataclass(frozen=True)
class RoomsSettings:
    rooms: tuple[Room, ...] = ()
    ups_minutes: float = 15
    ups_for_workstations: bool = False
    tariff_per_kwh: float = 8.0
    custom: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RoomsSettings":
        rooms = tuple(room for room in (room_from_mapping(item) for item in payload.get("rooms", []) if isinstance(item, Mapping)) if room)
        try:
            minutes = max(1.0, min(240.0, float(payload.get("ups_minutes", 15))))
        except (TypeError, ValueError):
            minutes = 15.0
        try:
            tariff = max(0.0, float(payload.get("tariff_per_kwh", 8.0)))
        except (TypeError, ValueError):
            tariff = 8.0
        return cls(rooms, minutes, _bool(payload.get("ups_for_workstations", False)), tariff, _bool(payload.get("custom", False)))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "rooms": [asdict(room) for room in self.rooms],
            "ups_minutes": self.ups_minutes,
            "ups_for_workstations": self.ups_for_workstations,
            "tariff_per_kwh": self.tariff_per_kwh,
            "custom": self.custom,
        }


class RoomsPresenter:
    def __init__(self, app_presenter: QtAppPresenter) -> None:
        self.app_presenter = app_presenter

    def saved_settings(self) -> RoomsSettings:
        return RoomsSettings.from_mapping(load_extra(self.app_presenter, SESSION_KEY))

    def save_settings(self, settings: RoomsSettings) -> None:
        save_extra(self.app_presenter, SESSION_KEY, settings.to_mapping())

    @staticmethod
    def default_rooms(components: Sequence[Mapping[str, Any]]) -> list[Room]:
        devices = count_devices(components)
        return rooms_from_counts(int(devices.workstations), int(devices.servers), int(devices.network_devices), int(devices.printers))

    def build(self, settings: RoomsSettings) -> PremisesEstimate:
        return estimate_premises(
            settings.rooms,
            RoomPrices(tariff_per_kwh=settings.tariff_per_kwh),
            ups_minutes=settings.ups_minutes,
            ups_for_workstations=settings.ups_for_workstations,
        )

    @staticmethod
    def room_rows(result: PremisesEstimate) -> list[dict[str, Any]]:
        return [
            {
                "name": item.room.name,
                "area": f"{item.room.area:.1f}".replace(".", ",") + " м²",
                "lan": f"{item.lan_cables} / {item.lan_meters:g} м",
                "climate": f"{item.required_cooling_w / 1000:.1f}".replace(".", ",") + f" кВт → {item.aircons} шт.",
                "light": f"{item.luminaires} шт., {item.achieved_lux} лк (норма {item.lux})",
                "total": money(item.total),
            }
            for item in result.rooms
        ]

    @staticmethod
    def line_rows(result: PremisesEstimate) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str, str, float], float] = {}
        for item in result.rooms:
            for line in item.lines:
                key = (line.group, line.name, line.unit, line.price)
                merged[key] = merged.get(key, 0) + line.quantity
        order = {name: index for index, name in enumerate(result.by_group)}
        rows = [
            {"group": group, "name": name, "quantity": f"{quantity:g} {unit}", "price": money(price), "total": money(quantity * price)}
            for (group, name, unit, price), quantity in sorted(merged.items(), key=lambda pair: order.get(pair[0][0], len(order)))
        ]
        if result.ups.model is not None:
            ups = result.ups
            packs = f" + {ups.packs} батарейн. блок(а)" if ups.packs else ""
            rows.append({"group": "ИБП", "name": f"ИБП {ups.model.va} ВА / {ups.model.watts} Вт{packs}", "quantity": "1 компл.", "price": money(ups.price), "total": money(ups.price)})
        return rows

    @staticmethod
    def summary(result: PremisesEstimate) -> str:
        if not result.rooms:
            return "Нет помещений: нажмите «По выбору» или добавьте помещение."
        groups = ", ".join(f"{name} {money(value)}" for name, value in result.by_group.items())
        ups = result.ups
        ups_text = (
            f" ИБП {ups.model.va} ВА держит {ups.load_w:g} Вт {str(ups.runtime_min).replace('.', ',')} мин."
            if ups.model
            else (" Нагрузка для ИБП не задана." if not ups.load_w else " Под нагрузку нет стандартного ИБП — нужен проект.")
        )
        return (
            f"Инфраструктура помещений: {money(result.total)} ({groups}).{ups_text} "
            f"Свет и кондиционеры: {str(round(result.monthly_kwh, 1)).replace('.', ',')} кВт·ч в месяц, {money(result.monthly_energy_cost)}."
        )

    def report_section(self, result: PremisesEstimate) -> dict[str, Any]:
        return {
            "paragraphs": [self.summary(result)],
            "tables": [
                {
                    "caption": "Таблица – Помещения: СКС, климат, освещение",
                    "header": ["Помещение", "Площадь", "Линии / кабель", "Охлаждение", "Освещение", "Сумма"],
                    "rows": [[r["name"], r["area"], r["lan"], r["climate"], r["light"], r["total"]] for r in self.room_rows(result)],
                },
                {
                    "caption": "Таблица – Смета инфраструктуры помещений",
                    "header": ["Раздел", "Позиция", "Кол-во", "Цена", "Сумма"],
                    "rows": [[r["group"], r["name"], r["quantity"], r["price"], r["total"]] for r in self.line_rows(result)],
                },
            ],
        }
