from tools.catalog_parser.legacy.dns_html_parsers import parse_product_html


def test_product_parser_reads_json_ld_offer_and_properties():
    html = """
    <html><body>
      <script type="application/ld+json">
        {
          "@type": "Product",
          "name": "Router JSON-LD",
          "offers": {"price": "7499"},
          "additionalProperty": [
            {"name": "Количество LAN портов", "value": "4"}
          ]
        }
      </script>
    </body></html>
    """

    product = parse_product_html(html, "https://example.test/router")

    assert product["title"] == "Router JSON-LD"
    assert product["price_int"] == 7499
    assert product["specs"]["Количество LAN портов"] == "4"
