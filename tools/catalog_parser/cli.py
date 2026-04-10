from __future__ import annotations

import argparse
from pathlib import Path

from .catalog_builder import save_catalog
from .paths import DEFAULT_OUTPUT_PATH
from .sources import build_catalog_from_example_snapshots, build_catalog_from_live_dns



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
        choices=["examples", "legacy-dns-live"],
        default="examples",
        help="Режим построения каталога. По умолчанию используется нормализация уже имеющихся example-снимков.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Путь, куда будет сохранён итоговый нормализованный каталог оборудования.",
    )
    return parser



def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)

    if args.mode == "examples":
        payload = build_catalog_from_example_snapshots()
    else:
        payload = build_catalog_from_live_dns()

    save_catalog(payload, output_path)
    print(f"Каталог оборудования сохранён: {output_path}")
    print(f"Записей в каталоге: {payload['stats']['items_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
