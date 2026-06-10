#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
APP_NAME="${APP_NAME:-ITCostCalc}"
DIST_ROOT="$ROOT/dist/linux"
BUILD_ROOT="$ROOT/build/pyinstaller-linux"
SPEC_ROOT="$ROOT/build/pyinstaller-specs"
APPDIR="$ROOT/build/appimage/${APP_NAME}.AppDir"
APPIMAGE_OUT="$DIST_ROOT/${APP_NAME}-x86_64.AppImage"

add_data_arg() {
  local source="$1"
  local target="$2"
  printf '%s:%s' "$source" "$target"
}

"$PYTHON_BIN" "$ROOT/scripts/doctor.py"

if ! "$PYTHON_BIN" -m PyInstaller --version >/dev/null 2>&1; then
  echo "ERROR: PyInstaller не установлен. Выполните: pip install -r requirements/build.txt" >&2
  exit 2
fi

rm -rf "$DIST_ROOT/$APP_NAME" "$BUILD_ROOT" "$APPDIR"
mkdir -p "$DIST_ROOT" "$SPEC_ROOT"

"$PYTHON_BIN" -m PyInstaller \
  --noconfirm \
  --clean \
  --onedir \
  --windowed \
  --name "$APP_NAME" \
  --paths "$ROOT/src" \
  --distpath "$DIST_ROOT" \
  --workpath "$BUILD_ROOT" \
  --specpath "$SPEC_ROOT" \
  --add-data "$(add_data_arg "$ROOT/data" "data")" \
  --add-data "$(add_data_arg "$ROOT/src/ui_qt/design" "ui_qt/design")" \
  --collect-data PySide6 \
  --collect-submodules PySide6 \
  --collect-data matplotlib \
  --hidden-import matplotlib.backends.backend_qtagg \
  "$ROOT/scripts/run_app.py"

if ! command -v appimagetool >/dev/null 2>&1; then
  echo "PyInstaller-сборка готова: $DIST_ROOT/$APP_NAME"
  echo "Для финального AppImage установите appimagetool и запустите скрипт повторно." >&2
  exit 0
fi

mkdir -p "$APPDIR/usr/bin"
cp -a "$DIST_ROOT/$APP_NAME/." "$APPDIR/usr/bin/"
install -m 755 "$ROOT/packaging/linux/AppRun" "$APPDIR/AppRun"
install -m 644 "$ROOT/packaging/linux/it-cost-calc.desktop" "$APPDIR/it-cost-calc.desktop"
install -m 644 "$ROOT/packaging/linux/it-cost-calc.svg" "$APPDIR/it-cost-calc.svg"

ARCH=x86_64 appimagetool "$APPDIR" "$APPIMAGE_OUT"
echo "Готово. AppImage: $APPIMAGE_OUT"
