from __future__ import annotations

__all__ = ["main"]


def __getattr__(name: str):
    if name != "main":
        raise AttributeError(name)
    from .dns_parser_legacy import main

    globals()[name] = main
    return main
