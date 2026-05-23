"""Источники для построения каталога оборудования.

Live DNS-парсер зависит от legacy-скрипта и внешних библиотек. Поэтому он
подключается лениво только при выборе режима ``legacy-dns-live``.
"""

from __future__ import annotations

from typing import Any

from .dns_examples import build_catalog_from_example_snapshots

__all__ = ["build_catalog_from_example_snapshots", "build_catalog_from_live_dns"]


def __getattr__(name: str) -> Any:
    if name == "build_catalog_from_live_dns":
        from .dns_live import build_catalog_from_live_dns

        globals()[name] = build_catalog_from_live_dns
        return build_catalog_from_live_dns

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
