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
    source_product_id: str | None = None
    identity: dict[str, Any] = field(default_factory=dict)
    offer: dict[str, Any] = field(default_factory=dict)
    field_provenance: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)

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
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
