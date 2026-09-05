from __future__ import annotations

import argparse
import os
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
        choices=[
            "examples",
            "dns-snapshot",
            "dns-har",
            "dns-html",
            "dns-live",
            "dns-http-live",
            "yandex-market-snapshot",
            "yandex-market-har",
            "yandex-market-html",
            "yandex-market-live",
            "yandex-market-http-live",
            "legacy-dns-live",
            "feed-download",
            "catalog-stage",
            "local-enrich",
            "icecat-enrich",
            "eis-benchmark",
            "commercial-quote",
            "browser-capture",
        ],
        default="examples",
        help="Режим построения каталога. По умолчанию используется нормализация уже имеющихся example-снимков.",
    )
    parser.add_argument(
        "--input",
        help="Каталог snapshot либо локальный HAR/HTML для соответствующего режима.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Путь, куда будет сохранён итоговый нормализованный каталог оборудования.",
    )
    parser.add_argument(
        "--categories",
        default="routers,switches,prebuilt_pcs,servers",
        help="Категории через запятую для live-режимов.",
    )
    parser.add_argument(
        "--browser-engine",
        choices=["firefox", "chromium"],
        default="firefox",
        help="Playwright-движок browser-live; при отсутствии устанавливается автоматически.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Карточек на категорию для live-режимов.")
    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        help="Общий таймаут live-сбора в секундах.",
    )
    parser.add_argument("--snapshot-output", help="Каталог диагностики/snapshot для live-сбора.")
    parser.add_argument("--profile", help="Каталог persistent browser profile для browser-live.")
    parser.add_argument("--region", default="", help="Текстовая метка региона цены.")
    parser.add_argument("--headless", action="store_true", help="Запустить browser-live без окна браузера.")
    parser.add_argument(
        "--browser-wait",
        type=float,
        default=8.0,
        help="Пауза на первой странице для региона/cookies/challenge.",
    )
    parser.add_argument("--feed-source-id", default="", help="Идентификатор структурированного feed-источника.")
    parser.add_argument("--feed-source-name", default="", help="Название структурированного feed-источника.")
    parser.add_argument(
        "--feed-format",
        choices=["auto", "xlsx", "csv", "xml", "yml"],
        default="auto",
        help="Формат структурированного feed.",
    )
    parser.add_argument(
        "--feed-price-kind",
        default="supplier_price",
        help="Семантика цены feed, например supplier_price.",
    )
    parser.add_argument(
        "--feed-download-strategy",
        choices=["direct", "yandex_disk_public"],
        default="direct",
        help="Способ получения feed по URL.",
    )
    parser.add_argument("--feed-manifest", help="Путь для provenance-манифеста feed.")
    parser.add_argument("--staging-path", help="Путь к catalog_staging.json для staging/enrichment.")
    parser.add_argument(
        "--staging-max-rows",
        type=int,
        default=0,
        help="Ограничить число строк staging; 0 означает обработать весь каталог.",
    )
    parser.add_argument("--local-manifest", help="Manifest автономного enrichment.")
    parser.add_argument(
        "--local-no-defaults",
        action="store_true",
        help="Не заполнять отсутствующие метрики демонстрационными defaults.",
    )
    parser.add_argument(
        "--local-no-estimated-price",
        action="store_true",
        help="Не строить демонстрационную цену для строк без реальной цены.",
    )
    parser.add_argument(
        "--staging-ids",
        default="",
        help="Staging ID через запятую; пусто означает все доступные позиции.",
    )
    parser.add_argument(
        "--icecat-username",
        default="",
        help="Логин Icecat; альтернативно ICECAT_USERNAME в окружении.",
    )
    parser.add_argument(
        "--icecat-language",
        default="EN",
        help="Язык Icecat feature names. Для встроенного mapping рекомендуется EN.",
    )
    parser.add_argument("--icecat-manifest", help="Диагностический manifest enrichment.")
    parser.add_argument(
        "--icecat-delay",
        type=float,
        default=0.15,
        help="Пауза между Icecat-запросами в секундах.",
    )
    parser.add_argument("--eis-manifest", help="Диагностический manifest применения ЕИС benchmark.")
    parser.add_argument(
        "--eis-max-records",
        type=int,
        default=20000,
        help="Максимум ценовых строк из XML/ZIP/JSON/CSV ЕИС.",
    )
    parser.add_argument("--quote-supplier", default="", help="Поставщик коммерческого предложения.")
    parser.add_argument("--quote-number", default="", help="Номер коммерческого предложения.")
    parser.add_argument("--quote-date", default="", help="Дата КП YYYY-MM-DD/ISO-8601.")
    parser.add_argument(
        "--quote-unknown-availability",
        action="store_true",
        help="Не считать позиции КП доступными по умолчанию.",
    )
    parser.add_argument("--capture-url", default="", help="URL страницы для browser capture provenance.")
    parser.add_argument(
        "--capture-category",
        default="",
        help="Явная категория единичного browser capture.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = Path(args.output)

    if args.mode == "catalog-stage":
        if not args.staging_path:
            parser.error("--staging-path is required for --mode catalog-stage")
        if not args.input:
            parser.error("--input is required for --mode catalog-stage")
        from application.services.catalog_staging_service import CatalogStagingService

        try:
            records = CatalogStagingService(Path(args.staging_path)).stage_file(
                args.input,
                max_rows=max(0, int(args.staging_max_rows)) or None,
                progress=lambda message: print(message, flush=True),
            )
        except (OSError, ValueError) as exc:
            print(f"Ошибка staging: {exc}", flush=True)
            return 11
        print(f"Staging готов: {len(records)} позиций.", flush=True)
        return 0

    if args.mode == "local-enrich":
        if not args.staging_path:
            parser.error("--staging-path is required for --mode local-enrich")
        from application.services.catalog_local_enrichment_service import (
            enrich_staging_records_locally,
        )
        from application.services.catalog_staging_service import CatalogStagingService

        staging_ids = tuple(
            value.strip() for value in args.staging_ids.split(",") if value.strip()
        )
        summary = enrich_staging_records_locally(
            CatalogStagingService(Path(args.staging_path)),
            staging_ids=staging_ids,
            fill_defaults=not args.local_no_defaults,
            estimate_missing_price=not args.local_no_estimated_price,
            manifest_path=args.local_manifest,
            progress=lambda message: print(message, flush=True),
        )
        result = summary.as_dict()
        print(
            "Автообогащение: "
            f"изменено {result['changed']}, явных полей {result['explicit_fields']}, "
            f"оценочных полей {result['default_fields']}, цен {result['estimated_prices']}.",
            flush=True,
        )
        return 0

    if args.mode == "eis-benchmark":
        if not args.staging_path:
            parser.error("--staging-path is required for --mode eis-benchmark")
        if not args.input:
            parser.error("--input is required for --mode eis-benchmark")
        from application.services.catalog_procurement_benchmark_service import (
            ProcurementBenchmarkError,
            apply_procurement_benchmarks,
            load_procurement_observations,
        )
        from application.services.catalog_staging_service import CatalogStagingService

        staging_ids = tuple(
            value.strip() for value in args.staging_ids.split(",") if value.strip()
        )
        try:
            observations = load_procurement_observations(
                args.input,
                region=args.region,
                max_records=max(1, args.eis_max_records),
                progress=lambda message: print(message, flush=True),
            )
            summary = apply_procurement_benchmarks(
                CatalogStagingService(Path(args.staging_path)),
                observations,
                staging_ids=staging_ids,
                source_location=args.input,
                region=args.region,
                manifest_path=args.eis_manifest,
                progress=lambda message: print(message, flush=True),
            )
        except ProcurementBenchmarkError as exc:
            print(f"Ошибка ЕИС benchmark: {exc}", flush=True)
            return 8
        result = summary.as_dict()
        print(
            "ЕИС benchmark: "
            f"наблюдений {result['observations']}, сопоставлено {result['matched_records']}, "
            f"по identity {result['identity_matches']}, по категории {result['category_matches']}.",
            flush=True,
        )
        return 0

    if args.mode == "commercial-quote":
        if not args.staging_path:
            parser.error("--staging-path is required for --mode commercial-quote")
        if not args.input:
            parser.error("--input is required for --mode commercial-quote")
        if not args.quote_supplier:
            parser.error("--quote-supplier is required for --mode commercial-quote")
        from application.services.catalog_commercial_quote_service import import_commercial_quote
        from application.services.catalog_staging_service import CatalogStagingService

        try:
            summary = import_commercial_quote(
                CatalogStagingService(Path(args.staging_path)),
                args.input,
                supplier_name=args.quote_supplier,
                quote_number=args.quote_number,
                quote_date=args.quote_date,
                region=args.region,
                assume_available=not args.quote_unknown_availability,
            )
        except ValueError as exc:
            print(f"Ошибка импорта КП: {exc}", flush=True)
            return 9
        print(
            f"КП импортировано: источник {summary.source_id}, строк {summary.records_total}.",
            flush=True,
        )
        return 0

    if args.mode == "browser-capture":
        if not args.staging_path:
            parser.error("--staging-path is required for --mode browser-capture")
        if not args.input:
            parser.error("--input is required for --mode browser-capture")
        from application.services.catalog_browser_capture_service import (
            BrowserCaptureError,
            capture_browser_file,
        )
        from application.services.catalog_staging_service import CatalogStagingService

        try:
            captured = capture_browser_file(
                args.input,
                source_url=args.capture_url,
                region=args.region,
                category_override=args.capture_category,
            )
            capture_path = Path(args.output)
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            capture_path.write_text(
                __import__("json").dumps(
                    {"schema_version": 2, "items": [captured.item]},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            CatalogStagingService(Path(args.staging_path)).stage_file(
                capture_path,
                source_context=captured.source_context,
            )
        except BrowserCaptureError as exc:
            print(f"Ошибка browser capture: {exc}", flush=True)
            return 10
        print(f"Browser capture добавлен: {capture_path}", flush=True)
        return 0

    if args.mode == "icecat-enrich":
        if not args.staging_path:
            parser.error("--staging-path is required for --mode icecat-enrich")
        from application.services.catalog_enrichment_service import (
            IcecatClient,
            IcecatConfigurationError,
            enrich_staging_records,
        )
        from application.services.catalog_staging_service import CatalogStagingService

        username = args.icecat_username or os.environ.get("ICECAT_USERNAME", "")
        token = os.environ.get("ICECAT_API_TOKEN", "")
        staging_ids = tuple(
            value.strip() for value in args.staging_ids.split(",") if value.strip()
        )
        try:
            client = IcecatClient(
                username=username,
                api_token=token,
                language=args.icecat_language,
            )
            summary = enrich_staging_records(
                CatalogStagingService(Path(args.staging_path)),
                client,
                staging_ids=staging_ids,
                manifest_path=args.icecat_manifest,
                request_delay_seconds=max(0.0, args.icecat_delay),
                progress=lambda message: print(message, flush=True),
            )
        except IcecatConfigurationError as exc:
            print(f"Ошибка доступа Icecat: {exc}", flush=True)
            return 7
        result = summary.as_dict()
        print(
            "Icecat: "
            f"совпало {result['matched']}, изменено {result['changed']}, "
            f"недоступно {result['unavailable']}, ошибок {result['errors']}.",
            flush=True,
        )
        if result["eligible"] and not result["matched"] and result["errors"]:
            return 6
        return 0

    if args.mode == "feed-download":
        if not args.input:
            parser.error("--input is required for --mode feed-download")
        from .feed_fetcher import CatalogFeedFetchError, fetch_catalog_feed

        try:
            result = fetch_catalog_feed(
                location=args.input,
                output_path=output_path,
                source_id=args.feed_source_id or Path(output_path).stem,
                source_name=args.feed_source_name or args.feed_source_id or Path(output_path).stem,
                feed_format=args.feed_format,
                region=args.region,
                price_kind=args.feed_price_kind,
                download_strategy=args.feed_download_strategy,
                manifest_path=args.feed_manifest,
            )
        except CatalogFeedFetchError as exc:
            print(f"Ошибка загрузки feed: {exc}", flush=True)
            return 6
        print(f"Feed сохранён: {result.output_path}", flush=True)
        print(f"Формат: {result.format}; байт: {result.size_bytes}", flush=True)
        if args.staging_path:
            from application.services.catalog_staging_service import CatalogStagingService

            print("Передаю feed в staging...", flush=True)
            records = CatalogStagingService(Path(args.staging_path)).stage_file(
                result.output_path,
                source_context=result.to_dict(),
                max_rows=max(0, int(args.staging_max_rows)) or None,
                progress=lambda message: print(message, flush=True),
            )
            print(f"Staging обновлён: {len(records)} позиций.", flush=True)
        return 0
    if args.mode == "examples":
        payload = build_catalog_from_example_snapshots()
    elif args.mode == "dns-snapshot":
        if not args.input:
            parser.error("--input is required for --mode dns-snapshot")
        from .sources.dns_snapshot import build_catalog_from_dns_snapshot

        payload = build_catalog_from_dns_snapshot(Path(args.input))
    elif args.mode in {"dns-har", "dns-html"}:
        if not args.input:
            parser.error(f"--input is required for --mode {args.mode}")
        from .sources.dns_capture import (
            DnsCaptureError,
            build_catalog_from_dns_har,
            build_catalog_from_dns_html,
        )

        try:
            if args.mode == "dns-har":
                payload = build_catalog_from_dns_har(Path(args.input), region=args.region)
            else:
                payload = build_catalog_from_dns_html(Path(args.input), region=args.region)
        except DnsCaptureError as exc:
            print(f"Ошибка локального DNS-import: {exc}", flush=True)
            return 5
    elif args.mode == "dns-live":
        if not args.snapshot_output or not args.profile:
            parser.error("--snapshot-output and --profile are required for --mode dns-live")
        from .dns_browser import DnsBrowserError
        from .sources.dns_live import (
            DnsLiveCollectionError,
            DnsLiveOptions,
            build_catalog_from_live_dns,
        )

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
                    browser_engine=args.browser_engine,
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
        except DnsLiveCollectionError as exc:
            print(f"Ошибка DNS-сбора: {exc}", flush=True)
            print(f"Диагностика: {exc.manifest_path}", flush=True)
            return exc.exit_code
        except DnsBrowserError as exc:
            print(f"Ошибка браузера DNS: {exc}", flush=True)
            return 4
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    elif args.mode == "dns-http-live":
        if not args.snapshot_output:
            parser.error("--snapshot-output is required for --mode dns-http-live")
        from .sources.dns_http_live import (
            DnsHttpCollectionError,
            DnsHttpLiveOptions,
            build_catalog_from_http_dns,
        )

        categories = tuple(value.strip() for value in args.categories.split(",") if value.strip())
        try:
            payload = build_catalog_from_http_dns(
                DnsHttpLiveOptions(
                    snapshot_dir=Path(args.snapshot_output),
                    categories=categories,
                    per_category_limit=args.limit,
                    time_limit_seconds=args.time_limit,
                    region=args.region,
                ),
                progress=lambda message: print(message, flush=True),
            )
        except KeyboardInterrupt:
            print("HTTP-сбор DNS остановлен пользователем.", flush=True)
            return 130
        except DnsHttpCollectionError as exc:
            print(f"Ошибка HTTP-сбора DNS: {exc}", flush=True)
            print(f"Диагностика: {exc.manifest_path}", flush=True)
            return exc.exit_code
    elif args.mode == "yandex-market-snapshot":
        if not args.input:
            parser.error("--input is required for --mode yandex-market-snapshot")
        from .sources.yandex_market_snapshot import build_catalog_from_yandex_market_snapshot

        payload = build_catalog_from_yandex_market_snapshot(Path(args.input))
    elif args.mode in {"yandex-market-har", "yandex-market-html"}:
        if not args.input:
            parser.error(f"--input is required for --mode {args.mode}")
        from .sources.yandex_market_capture import (
            YandexMarketCaptureError,
            build_catalog_from_yandex_market_har,
            build_catalog_from_yandex_market_html,
        )

        try:
            if args.mode == "yandex-market-har":
                payload = build_catalog_from_yandex_market_har(Path(args.input), region=args.region)
            else:
                payload = build_catalog_from_yandex_market_html(Path(args.input), region=args.region)
        except YandexMarketCaptureError as exc:
            print(f"Ошибка локального импорта Яндекс Маркета: {exc}", flush=True)
            return 5
    elif args.mode == "yandex-market-live":
        if not args.snapshot_output or not args.profile:
            parser.error(
                "--snapshot-output and --profile are required for --mode yandex-market-live"
            )
        from .market_browser import YandexMarketBrowserError
        from .sources.yandex_market_live import (
            YandexMarketLiveCollectionError,
            YandexMarketLiveOptions,
            build_catalog_from_live_yandex_market,
        )

        categories = tuple(value.strip() for value in args.categories.split(",") if value.strip())
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def cancel_market_collection(_signum, _frame) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, cancel_market_collection)
        try:
            payload = build_catalog_from_live_yandex_market(
                YandexMarketLiveOptions(
                    snapshot_dir=Path(args.snapshot_output),
                    profile_dir=Path(args.profile),
                    categories=categories,
                    browser_engine=args.browser_engine,
                    per_category_limit=args.limit,
                    time_limit_seconds=args.time_limit,
                    headless=args.headless,
                    first_page_wait_seconds=args.browser_wait,
                    region=args.region,
                ),
                progress=lambda message: print(message, flush=True),
            )
        except KeyboardInterrupt:
            print("Сбор Яндекс Маркета остановлен пользователем.", flush=True)
            return 130
        except YandexMarketLiveCollectionError as exc:
            print(f"Ошибка сбора Яндекс Маркета: {exc}", flush=True)
            print(f"Диагностика: {exc.manifest_path}", flush=True)
            return exc.exit_code
        except YandexMarketBrowserError as exc:
            print(f"Ошибка браузера Яндекс Маркета: {exc}", flush=True)
            return 4
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    elif args.mode == "yandex-market-http-live":
        if not args.snapshot_output:
            parser.error("--snapshot-output is required for --mode yandex-market-http-live")
        from .sources.yandex_market_http_live import (
            YandexMarketHttpCollectionError,
            YandexMarketHttpLiveOptions,
            build_catalog_from_http_yandex_market,
        )

        categories = tuple(value.strip() for value in args.categories.split(",") if value.strip())
        try:
            payload = build_catalog_from_http_yandex_market(
                YandexMarketHttpLiveOptions(
                    snapshot_dir=Path(args.snapshot_output),
                    categories=categories,
                    per_category_limit=args.limit,
                    time_limit_seconds=args.time_limit,
                    region=args.region,
                ),
                progress=lambda message: print(message, flush=True),
            )
        except KeyboardInterrupt:
            print("HTTP-сбор Яндекс Маркета остановлен пользователем.", flush=True)
            return 130
        except YandexMarketHttpCollectionError as exc:
            print(f"Ошибка HTTP-сбора Яндекс Маркета: {exc}", flush=True)
            print(f"Диагностика: {exc.manifest_path}", flush=True)
            return exc.exit_code
    else:
        from .sources.dns_live import build_catalog_from_legacy_dns

        payload = build_catalog_from_legacy_dns()

    save_catalog(payload, output_path)
    print(f"Каталог оборудования сохранён: {output_path}")
    print(f"Записей в каталоге: {payload['stats']['items_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
