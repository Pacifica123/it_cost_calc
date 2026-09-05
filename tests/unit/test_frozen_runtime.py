from pathlib import Path
import sys

from shared import runtime
from ui_qt.presenters import app_presenter


def test_frozen_catalog_parser_relaunches_application(monkeypatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/opt/ITCostCalc/ITCostCalc")

    program, arguments = runtime.catalog_parser_process(Path("/ignored"))

    assert program == "/opt/ITCostCalc/ITCostCalc"
    assert arguments == ("--catalog-parser",)


def test_frozen_playwright_install_relaunches_application(monkeypatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/opt/ITCostCalc/ITCostCalc")

    assert runtime.playwright_install_command("firefox") == [
        "/opt/ITCostCalc/ITCostCalc",
        "--playwright-install",
        "firefox",
    ]


def test_appimage_external_environment_restores_host_library_path(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    env = runtime.external_process_environment(
        {
            "APPIMAGE": "/tmp/ITCostCalc.AppImage",
            "LD_LIBRARY_PATH": "/tmp/.mount/app/lib",
            "LD_LIBRARY_PATH_ORIG": "/usr/local/lib",
            "PYTHONHOME": "/tmp/.mount/app",
            "PYTHONPATH": "/tmp/.mount/app/python",
            "PATH": "/usr/bin",
        }
    )

    assert env["LD_LIBRARY_PATH"] == "/usr/local/lib"
    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env
    assert env["PATH"] == "/usr/bin"


def test_frozen_writable_root_uses_platform_user_data(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    assert runtime.writable_runtime_root() == (tmp_path / "ITCostCalc").resolve()


def test_default_qt_paths_split_writable_data_from_bundled_fixtures(
    monkeypatch, tmp_path: Path
) -> None:
    writable = tmp_path / "writable"
    bundled = tmp_path / "bundle"
    monkeypatch.setattr(app_presenter, "writable_runtime_root", lambda: writable)
    monkeypatch.setattr(app_presenter, "resource_root", lambda: bundled)

    paths = app_presenter.QtRuntimePaths.from_repo_root()

    assert paths.repo_root == writable
    assert paths.resource_root == bundled
    assert paths.runtime_entities_path == writable / "data/generated/runtime_entities.json"
    assert paths.demo_dataset_path == bundled / "data/fixtures/demo_dataset.json"
    assert paths.demo_profiles_path == bundled / "data/fixtures/demo_profiles.json"


def test_frozen_playwright_cache_uses_user_cache_instead_of_bundle(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    env = {
        "PLAYWRIGHT_BROWSERS_PATH": "0",
        "XDG_CACHE_HOME": str(tmp_path),
    }

    path = runtime.configure_playwright_environment(env)

    assert path == (tmp_path / "ms-playwright").resolve()
    assert env["PLAYWRIGHT_BROWSERS_PATH"] == str(path)


def test_frozen_playwright_cache_keeps_explicit_user_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    custom = tmp_path / "custom-playwright"
    env = {"PLAYWRIGHT_BROWSERS_PATH": str(custom)}

    path = runtime.configure_playwright_environment(env)

    assert path == custom.resolve()
    assert env["PLAYWRIGHT_BROWSERS_PATH"] == str(custom)
