from __future__ import annotations

import argparse
import signal
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
        choices=["examples", "dns-snapshot", "dns-live", "legacy-dns-live"],
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
    parser.add_argument(
        "--categories",
        default="routers,prebuilt_pcs,servers",
        help="DNS-категории через запятую для режима dns-live.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Карточек на категорию для dns-live.")
    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        help="Общий таймаут dns-live в секундах.",
    )
    parser.add_argument("--snapshot-output", help="Каталог для HTML-снимка dns-live.")
    parser.add_argument("--profile", help="Каталог persistent Chromium profile для dns-live.")
    parser.add_argument("--region", default="", help="Текстовая метка региона цены для dns-live.")
    parser.add_argument("--headless", action="store_true", help="Запустить dns-live без окна браузера.")
    parser.add_argument(
        "--browser-wait",
        type=float,
        default=8.0,
        help="Пауза на первой странице для региона/cookies/challenge.",
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
    elif args.mode == "dns-live":
        if not args.snapshot_output or not args.profile:
            parser.error("--snapshot-output and --profile are required for --mode dns-live")
        from .sources.dns_live import DnsLiveOptions, build_catalog_from_live_dns

        categories = tuple(value.strip() for value in args.categories.split(",") if value.strip())
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def cancel_live_collection(_signum, _frame) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, cancel_live_collection)
        try:
            payload = build_catalog_from_live_dns(
                DnsLiveOptions(
                    snapshot_dir=Path(args.snapshot_output),
                    profile_dir=Path(args.profile),
                    categories=categories,
                    per_category_limit=args.limit,
                    time_limit_seconds=args.time_limit,
                    headless=args.headless,
                    first_page_wait_seconds=args.browser_wait,
                    region=args.region,
                ),
                progress=lambda message: print(message, flush=True),
            )
        except KeyboardInterrupt:
            print("Сбор DNS остановлен пользователем.", flush=True)
            return 130
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    else:
        from .sources.dns_live import build_catalog_from_legacy_dns

        payload = build_catalog_from_legacy_dns()

    save_catalog(payload, output_path)
    print(f"Каталог оборудования сохранён: {output_path}")
    print(f"Записей в каталоге: {payload['stats']['items_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
