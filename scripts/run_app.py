from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from it_cost_calc.bootstrap import main as app_main

    app_main()


if __name__ == "__main__":
    main()
