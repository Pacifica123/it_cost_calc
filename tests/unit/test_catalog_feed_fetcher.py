from __future__ import annotations

import json
from pathlib import Path

from tools.catalog_parser.feed_fetcher import CatalogFeedFetchError, fetch_catalog_feed


def test_local_feed_copy_writes_provenance_manifest(tmp_path: Path):
    source = tmp_path / "catalog.csv"
    source.write_text("name;price\nRouter;1000\n", encoding="utf-8")
    output = tmp_path / "run" / "supplier.csv"
    manifest = tmp_path / "run" / "fetch_manifest.json"

    result = fetch_catalog_feed(
        location=str(source),
        output_path=output,
        source_id="supplier",
        source_name="Supplier",
        feed_format="csv",
        region="Россия",
        price_kind="supplier_price",
        manifest_path=manifest,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert output.read_bytes() == source.read_bytes()
    assert result.format == "csv"
    assert payload["source_id"] == "supplier"
    assert payload["observed_at"]
    assert len(payload["sha256"]) == 64


def test_expected_csv_rejects_html_error_page(tmp_path: Path):
    source = tmp_path / "blocked.csv"
    source.write_text("<!doctype html><html><body>captcha</body></html>", encoding="utf-8")

    try:
        fetch_catalog_feed(
            location=str(source),
            output_path=tmp_path / "out.csv",
            source_id="blocked",
            source_name="Blocked",
            feed_format="csv",
        )
    except CatalogFeedFetchError as exc:
        assert "HTML" in str(exc)
    else:
        raise AssertionError("HTML response must not be accepted as CSV feed")
