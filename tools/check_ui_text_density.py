"""Static text-density check for desktop UI code.

The migration contract forbids long visible labels in the UI.  This script
performs a conservative AST-based scan for literal texts passed to common
Tkinter/ttk and PySide6/Qt widget APIs.  Tooltip/help/details strings are
intentionally excluded because long explanations should live there.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_MAX_WORDS = 10
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-–—][A-Za-zА-Яа-яЁё0-9]+)?")

# API names whose literal string arguments are normally visible on screen.
VISIBLE_CALL_NAMES = {
    # Tk / ttk widgets
    "Label",
    "Button",
    "LabelFrame",
    "Checkbutton",
    "Radiobutton",
    "Menubutton",
    # Qt widgets/actions
    "QLabel",
    "QPushButton",
    "QToolButton",
    "QCheckBox",
    "QRadioButton",
    "QGroupBox",
    "QAction",
    "QMenu",
    # Project-level future helpers
    "CompactLabel",
    "PrimaryButton",
    "SecondaryButton",
    "DangerButton",
    "EmptyState",
    "ActionBar",
    "StatusStrip",
    "CollapsibleSection",
    "SettingsPanel",
    "SmartTable",
    "RecordFormDialog",
    "FieldSpec",
}

VISIBLE_METHOD_NAMES = {
    "setText",
    "setTitle",
    "setWindowTitle",
    "setPlaceholderText",
    "set_message",
    "set_status",
    "add_primary_action",
    "add_secondary_action",
    "addTab",
    "insertTab",
    "setTabText",
    "addAction",
    "addMenu",
    "title",
}

VISIBLE_KEYWORDS = {
    "text",
    "title",
    "label",
    "placeholder",
    "placeholder_text",
    "window_title",
    "message",
    "status",
    "subtitle",
    "description",
    "action_text",
    "empty_title",
    "empty_status",
    "add_text",
    "more_text",
}

HELP_OR_TOOLTIP_NAMES = {
    "ToolTip",
    "Tooltip",
    "InfoHint",
    "HelpText",
    "HelpPanel",
    "DetailsPanel",
    "attach_tooltip",
    "setToolTip",
    "setStatusTip",
    "setWhatsThis",
}

SKIPPED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
    "node_modules",
    "data/generated",
}


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    words: int
    source: str
    text: str


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_help_or_tooltip(name: str) -> bool:
    lowered = name.lower()
    return name in HELP_OR_TOOLTIP_NAMES or "tooltip" in lowered or "help" in lowered


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text.replace("\n", " ")))


def _string_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None
        return "".join(parts)
    return None


def _iter_python_files(paths: Sequence[Path]) -> Iterable[Path]:
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.exists():
            continue
        if path.is_file() and path.suffix == ".py":
            yield path
            continue
        if not path.is_dir():
            continue
        for candidate in path.rglob("*.py"):
            parts = set(candidate.relative_to(path).parts)
            if parts & SKIPPED_DIRS:
                continue
            yield candidate


def _check_text(
    *,
    path: Path,
    line: int,
    source: str,
    text: str,
    max_words: int,
) -> Violation | None:
    normalized = " ".join(text.split())
    if not normalized:
        return None
    words = _word_count(normalized)
    if words <= max_words:
        return None
    return Violation(path=path, line=line, words=words, source=source, text=normalized)


def scan_file(path: Path, *, max_words: int) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations: list[Violation] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        name = _call_name(node.func)
        if _is_help_or_tooltip(name):
            continue

        is_visible_call = name in VISIBLE_CALL_NAMES or name in VISIBLE_METHOD_NAMES
        if not is_visible_call:
            continue

        for index, arg in enumerate(node.args):
            text = _string_value(arg)
            if text is None:
                continue
            violation = _check_text(
                path=path,
                line=getattr(arg, "lineno", getattr(node, "lineno", 0)),
                source=f"{name} arg#{index + 1}",
                text=text,
                max_words=max_words,
            )
            if violation is not None:
                violations.append(violation)

        for keyword in node.keywords:
            if keyword.arg not in VISIBLE_KEYWORDS:
                continue
            text = _string_value(keyword.value)
            if text is None:
                continue
            violation = _check_text(
                path=path,
                line=getattr(keyword.value, "lineno", getattr(node, "lineno", 0)),
                source=f"{name} {keyword.arg}=",
                text=text,
                max_words=max_words,
            )
            if violation is not None:
                violations.append(violation)

    return violations


def _default_scan_paths(root: Path) -> list[Path]:
    candidates = [root / "src" / "ui_qt"]
    return [path for path in candidates if path.exists()]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that visible UI texts stay within the migration density limit.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to scan. Defaults to src/ui_qt.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=DEFAULT_MAX_WORDS,
        help=f"Maximum words in visible text. Default: {DEFAULT_MAX_WORDS}.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = Path.cwd()
    scan_paths = args.paths or _default_scan_paths(root)

    if not scan_paths:
        print("UI text density: no UI directories found; nothing to scan.")
        return 0

    violations: list[Violation] = []
    for path in sorted(set(_iter_python_files(scan_paths))):
        try:
            violations.extend(scan_file(path, max_words=args.max_words))
        except SyntaxError as exc:
            print(f"UI text density: syntax error in {path}: {exc}", file=sys.stderr)
            return 2

    if not violations:
        scanned = ", ".join(str(path) for path in scan_paths)
        print(f"UI text density: OK, visible texts are <= {args.max_words} words ({scanned}).")
        return 0

    print(f"UI text density: found {len(violations)} long visible text(s):", file=sys.stderr)
    for item in violations:
        display_text = item.text if len(item.text) <= 140 else item.text[:137] + "..."
        print(
            f"- {item.path}:{item.line}: {item.words} words in {item.source}: {display_text!r}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
