from __future__ import annotations

import argparse
from pathlib import Path

from .catalog_builder import save_catalog
from .paths import DEFAULT_OUTPUT_PATH
from .sources.dns_examples import build_catalog_from_example_snapshots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="catalog-parser",
        description=(
            "Собирает или нормализует каталог оборудования как отдельный вспомогательный инструмент "
            "вне runtime основного GUI-приложения."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["examples", "dns-snapshot", "legacy-dns-live"],
        default="examples",
        help="Режим построения каталога. По умолчанию используется нормализация уже имеющихся example-снимков.",
    )
    parser.add_argument(
        "--input",
        help="Каталог локального DNS-снимка. Обязателен для режима dns-snapshot.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Путь, куда будет сохранён итоговый нормализованный каталог оборудования.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = Path(args.output)

    if args.mode == "examples":
        payload = build_catalog_from_example_snapshots()
    elif args.mode == "dns-snapshot":
        if not args.input:
            parser.error("--input is required for --mode dns-snapshot")
        from .sources.dns_snapshot import build_catalog_from_dns_snapshot

        payload = build_catalog_from_dns_snapshot(Path(args.input))
    else:
        from .sources.dns_live import build_catalog_from_live_dns

        payload = build_catalog_from_live_dns()

    save_catalog(payload, output_path)
    print(f"Каталог оборудования сохранён: {output_path}")
    print(f"Записей в каталоге: {payload['stats']['items_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
