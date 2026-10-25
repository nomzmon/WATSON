"""Export a blind human rating sheet and its answer key from a run.

Writes rating_sheet.csv (for the raters) and answer_key.csv (keep it from them) to
experiments/<experiment>/human_ratings/<date>/. Raters copy the sheet to rated_<their name>.csv in that folder
and fill in the persona and narrative columns, following
evaluation_instruments/annotation/reply_rating_guide.md.
Usage (from the repo root):
    python scripts/export_rating_sheet.py experiments/framework_validation/runs/<run> --per-step 10
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from watson.common.config import resolve_path
from watson.evaluation.human_ratings import select_items, write_sheet_and_key
from watson.evaluation.middleware_regression import load_turn_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="run folder containing a turns.csv")
    parser.add_argument("--per-step", type=int, default=10, help="replies to take from each step (default: 10)")
    parser.add_argument("--seed", type=int, default=7, help="seed for choosing and shuffling replies (default: 7)")
    args = parser.parse_args()

    run = resolve_path(Path(args.run))
    items = select_items(load_turn_records(run / "turns.csv"), args.per_step, args.seed)
    folder = run.parent.parent / f"human_ratings/{datetime.now():%Y-%m-%d_%H%M}"
    sheet, key = write_sheet_and_key(items, folder)
    steps = sorted({item.step for item in items})
    print(f"{len(items)} replies from steps {', '.join(map(str, steps))}.")
    print(f"Rating sheet (give to raters): {sheet}")
    print(f"Answer key (keep from raters): {key}")


if __name__ == "__main__":
    main()