from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .dns_compatibility import compatible_pair
from .dns_constants import (
    BASE_URL,
    CATEGORIES,
    GENERATED_BUILDS_COUNT,
    MAX_JSON_BYTES,
    OUTPUT_FILE,
    PER_CATEGORY_LIMIT,
    TOTAL_TIME_LIMIT_SECONDS,
)
from .dns_fetcher import build_fetcher
from .dns_html_parsers import parse_product_html, parse_search_html

logger = logging.getLogger(__name__)


def main() -> None:
    start_time = time.time()
    fetch, close_browser = build_fetcher()
    logger.info("Запущен сборщик каталога DNS")
    try:
        collected = {
            "routers": [],
            "prebuilt_pcs": [],
            "components": [],
            "compatible_builds": [],
        }

        for key, query in CATEGORIES.items():
            if time.time() - start_time > TOTAL_TIME_LIMIT_SECONDS:
                logger.warning("Превышен лимит времени сбора каталога")
                break
            logger.info("Сбор категории %s -> запрос '%s'", key, query)
            query_url = f"{BASE_URL}/search/?q={query}"
            html = fetch(query_url, wait=2.0)
            if not html:
                logger.warning("Пустой HTML для запроса '%s'", query)
                continue
            items = parse_search_html(html, limit=PER_CATEGORY_LIMIT)
            logger.info("По категории %s найдено ~%s товаров", key, len(items))
            bucket: List[Dict] = []
            for item in tqdm(items, desc=f"parsing {key}", leave=False):
                if time.time() - start_time > TOTAL_TIME_LIMIT_SECONDS:
                    logger.warning("Превышен лимит времени во время парсинга %s", key)
                    break
                url = item.get("url")
                if not url:
                    continue
                product_html = fetch(url, wait=0.6)
                if not product_html:
                    continue
                product = parse_product_html(product_html, url)
                bucket.append(product)
                if key in (
                    "motherboards",
                    "cpus",
                    "gpus",
                    "rams",
                    "psus",
                    "ssds",
                    "hdds",
                    "cases",
                ):
                    product["type"] = key
                    collected["components"].append(product)
                elif key == "routers":
                    collected["routers"].append(product)
                elif key == "prebuilt_pcs":
                    collected["prebuilt_pcs"].append(product)
                else:
                    product["type"] = key
                    collected["components"].append(product)
            logger.debug("Категория %s обработана, карточек=%s", key, len(bucket))
            time.sleep(0.8)

        logger.info("Генерация локально совместимых сборок")
        cpus = [c for c in collected["components"] if c.get("type") == "cpus"]
        motherboards = [c for c in collected["components"] if c.get("type") == "motherboards"]
        rams = [c for c in collected["components"] if c.get("type") == "rams"]
        psus = [c for c in collected["components"] if c.get("type") == "psus"]
        cases = [c for c in collected["components"] if c.get("type") == "cases"]
        gpus = [c for c in collected["components"] if c.get("type") == "gpus"]
        storages = [
            c for c in collected["components"] if c.get("type") in ("ssds", "hdds")
        ]

        generation_limit = min(GENERATED_BUILDS_COUNT, max(1, len(cpus), len(motherboards), len(rams)))
        tries = 0
        builds = []
        while len(builds) < generation_limit and tries < generation_limit * 50:
            tries += 1
            cpu = random.choice(cpus) if cpus else None
            motherboard = random.choice(motherboards) if motherboards else None
            ram = random.choice(rams) if rams else None
            psu = random.choice(psus) if psus else None
            case = random.choice(cases) if cases else None
            gpu = random.choice(gpus) if gpus else None
            storage = random.choice(storages) if storages else None
            if not (cpu and motherboard and ram and psu and case):
                break
            if compatible_pair(cpu, motherboard, ram, psu, case):
                total_price = sum(
                    [
                        cpu.get("price_int", 0),
                        motherboard.get("price_int", 0),
                        ram.get("price_int", 0),
                        psu.get("price_int", 0),
                        (gpu.get("price_int", 0) if gpu else 0),
                        (storage.get("price_int", 0) if storage else 0),
                        case.get("price_int", 0),
                    ]
                )
                builds.append(
                    {
                        "cpu": cpu,
                        "motherboard": motherboard,
                        "ram": ram,
                        "psu": psu,
                        "case": case,
                        "gpu": gpu,
                        "storage": storage,
                        "total_price": total_price,
                    }
                )
        collected["compatible_builds"] = builds

        logger.info("Сериализация и проверка размера JSON")

        def serialize_and_shrink(data: Dict[str, List[Dict]]):
            blob = json.dumps(data, ensure_ascii=False).encode("utf-8")
            if len(blob) <= MAX_JSON_BYTES:
                return blob, data
            logger.warning("JSON превысил лимит размера: %s bytes. Запускается shrink.", len(blob))
            data_copy = data.copy()
            components = data_copy.get("components", [])
            routers = data_copy.get("routers", [])
            prebuilt = data_copy.get("prebuilt_pcs", [])
            builds_local = data_copy.get("compatible_builds", [])
            target = MAX_JSON_BYTES
            factor = 0.6
            while True:
                new_components_len = max(10, int(len(components) * factor))
                new_builds_len = max(1, int(len(builds_local) * factor))
                new_data = {
                    "routers": routers[: max(10, int(len(routers) * factor))],
                    "prebuilt_pcs": prebuilt[: max(5, int(len(prebuilt) * factor))],
                    "components": components[:new_components_len],
                    "compatible_builds": builds_local[:new_builds_len],
                }
                blob = json.dumps(new_data, ensure_ascii=False).encode("utf-8")
                if len(blob) <= target or factor < 0.05:
                    return blob, new_data
                factor *= 0.7

        blob, final_data = serialize_and_shrink(collected)
        Path(OUTPUT_FILE).write_bytes(blob)
        logger.info(
            "Сохранён каталог DNS: routers=%s, prebuilt=%s, components=%s, builds=%s, file=%s, size=%s bytes",
            len(final_data.get("routers", [])),
            len(final_data.get("prebuilt_pcs", [])),
            len(final_data.get("components", [])),
            len(final_data.get("compatible_builds", [])),
            OUTPUT_FILE,
            len(blob),
        )
    finally:
        close_browser()
