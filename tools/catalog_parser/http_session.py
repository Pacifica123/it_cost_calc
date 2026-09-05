from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse

import requests

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class CatalogHttpRequestError(RuntimeError):
    """Network error raised by the lightweight HTTP catalog transport."""


@dataclass(frozen=True, slots=True)
class CatalogHttpResponse:
    requested_url: str
    final_url: str
    status_code: int
    text: str
    content_type: str


class CatalogHttpSession:
    """Small cookie-preserving HTTP client used by non-Playwright collectors.

    It deliberately does not attempt to bypass CAPTCHA/challenge systems.  The
    session only keeps normal cookies returned by the public storefront and
    uses ordinary browser-like request headers.
    """

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": _DEFAULT_USER_AGENT,
                "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.7",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }
        )

    @property
    def cookies(self) -> Mapping[str, str]:
        return self._session.cookies.get_dict()

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "CatalogHttpSession":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        xhr: bool = False,
        referer: str | None = None,
    ) -> CatalogHttpResponse:
        headers = self._request_headers(url, xhr=xhr, referer=referer)
        if xhr:
            headers["Accept"] = "application/json, text/plain, */*"
        else:
            headers["Accept"] = (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,*/*;q=0.8"
            )
        return self._request("GET", url, params=params, headers=headers)

    def post_form(
        self,
        url: str,
        *,
        data: str,
        referer: str | None = None,
        csrf_token: str | None = None,
    ) -> CatalogHttpResponse:
        headers = self._request_headers(url, xhr=True, referer=referer)
        headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            }
        )
        if csrf_token:
            headers["X-CSRF-Token"] = csrf_token
        return self._request("POST", url, data=data, headers=headers)

    def _request_headers(
        self,
        url: str,
        *,
        xhr: bool,
        referer: str | None,
    ) -> dict[str, str]:
        parsed = urlparse(url)
        headers: dict[str, str] = {
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "cors" if xhr else "navigate",
            "Sec-Fetch-Dest": "empty" if xhr else "document",
        }
        if xhr:
            headers["X-Requested-With"] = "XMLHttpRequest"
        if referer:
            headers["Referer"] = referer
        if parsed.scheme and parsed.netloc:
            headers["Origin"] = f"{parsed.scheme}://{parsed.netloc}"
        return headers

    def _request(self, method: str, url: str, **kwargs) -> CatalogHttpResponse:
        try:
            response = self._session.request(
                method,
                url,
                timeout=(10.0, self.timeout_seconds),
                allow_redirects=True,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise CatalogHttpRequestError(f"HTTP-запрос не выполнен: {exc}") from exc
        return CatalogHttpResponse(
            requested_url=url,
            final_url=str(response.url),
            status_code=int(response.status_code),
            text=response.text,
            content_type=str(response.headers.get("content-type") or ""),
        )
