from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CatalogItem:
    """Нормализованная запись каталога оборудования."""

    item_id: str
    title: str
    category: str
    source: str
    source_category: str
    price_rub: int | None
    currency: str
    availability: str
    url: str | None
    attributes: dict[str, Any] = field(default_factory=dict)
    raw_snapshot: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CatalogSourceInfo:
    """Метаданные об источнике, из которого был собран каталог."""

    source: str
    snapshot_name: str
    mode: str
    items_before_dedup: int
    items_after_dedup: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
