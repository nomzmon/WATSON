"""Combine repeated runs of an experiment (different seeds) into one table of averages.

Give the run folders (each holds a turns.csv). The report is written to the experiment's reports folder.
Usage (from the repo root):
    python scripts/combine_runs.py experiments/framework_validation/runs/<run1> experiments/framework_validation/runs/<run2>
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from watson.common.config import resolve_path
from watson.evaluation.middleware_regression import load_turn_records
from watson.evaluation.seed_summary import combine_runs, render_combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="run folders, each containing a turns.csv")
    args = parser.parse_args()

    runs = [load_turn_records(resolve_path(Path(run) / "turns.csv")) for run in args.runs]
    started = datetime.now()
    report = render_combined(started, args.runs, combine_runs(runs))
    print(report)

    report_path = resolve_path(Path(args.runs[0])).parent.parent / f"reports/{started:%Y-%m-%d_%H%M}_combined.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()