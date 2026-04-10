from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from bootstrap import load_demo_data
    from infrastructure.logging import configure_logging

    configure_logging(repo_root=ROOT)
    dataset = load_demo_data(ROOT)
    total_rows = sum(len(rows) for rows in dataset.values())
    print(f"Демо-набор загружен в runtime-хранилище: {total_rows} строк")


if __name__ == "__main__":
    main()
