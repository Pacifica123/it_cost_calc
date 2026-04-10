"""Вспомогательный инструмент построения каталога оборудования.

Каталог `tools/catalog_parser/` намеренно отделён от основного runtime GUI-приложения.
Он нужен для подготовки и обновления каталога оборудования, который в будущем сможет
использоваться приложением как источник справочных данных.
"""

from .catalog_builder import build_catalog_payload, deduplicate_items, normalize_dns_snapshot, save_catalog
from .cli import main

__all__ = [
    "build_catalog_payload",
    "deduplicate_items",
    "normalize_dns_snapshot",
    "save_catalog",
    "main",
]
