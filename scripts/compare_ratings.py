"""Compare filled-in human rating sheets with the judges' scores.

Reads answer_key.csv and every rated_<name>.csv in a human_ratings folder made by export_rating_sheet.py and
writes comparison.md there.
Usage (from the repo root):
    python scripts/compare_ratings.py experiments/framework_validation/human_ratings/<date>
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from watson.common.config import resolve_path
from watson.evaluation.human_ratings import read_key, read_ratings, render_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="human_ratings folder with answer_key.csv and rated_<name>.csv files")
    args = parser.parse_args()

    folder = resolve_path(Path(args.folder))
    key = read_key(folder / "answer_key.csv")
    items = [row["item"] for row in key]
    raters = {path.stem.removeprefix("rated_"): read_ratings(path, items) for path in sorted(folder.glob("rated_*.csv"))}
    if not raters:
        parser.error(f"no rated_<name>.csv files in {folder}")
    report = render_comparison(datetime.now(), key, raters)
    print(report)
    (folder / "comparison.md").write_text(report, encoding="utf-8")
    print(f"Report: {folder / 'comparison.md'}")


if __name__ == "__main__":
    main()