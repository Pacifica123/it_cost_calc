"""Compatibility wrapper for the refactored legacy DNS parser."""

try:
    from .dns_workflow import main
except Exception:  # pragma: no cover - direct script execution fallback
    from dns_workflow import main


if __name__ == "__main__":
    main()
