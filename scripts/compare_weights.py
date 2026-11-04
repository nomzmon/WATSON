"""Compare the current rubric weights with a variant (default: equal weights) on finished runs.

Recomputes every input decision (WICS), guidance decision (DFMS) and first-reply validation result (OVS) from the
judges' criterion scores in the runs' turns.csv, under both sets of weights; no model is called. The variant
weights come from configs/rubrics/variants/{wics,dfms,ovs}_<variant>.yaml. The report is written to the
experiment's reports folder.
Usage (from the repo root):
    python scripts/compare_weights.py experiments/long_conversation/runs/<run1> experiments/long_conversation/runs/<run2>
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from watson.common.config import resolve_path
from watson.common.rubric import load_rubric_weights
from watson.evaluation.middleware_regression import load_turn_records
from watson.evaluation.weight_comparison import compare_weights, render_weight_comparison
from watson.middleware.dialogue_flow.flow_evaluator import DFMS_CRITERIA
from watson.middleware.input_constraint.scoring import WICS_CRITERIA
from watson.middleware.output_validation.validator import OVS_CRITERIA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="run folders, each containing a turns.csv")
    parser.add_argument("--variant", default="equal", help="variant name in configs/rubrics/variants (default: equal)")
    args = parser.parse_args()

    variants = {name: load_rubric_weights(f"configs/rubrics/variants/{name.lower()}_{args.variant}.yaml", criteria, name)
                for name, criteria in (("WICS", WICS_CRITERIA), ("DFMS", DFMS_CRITERIA), ("OVS", OVS_CRITERIA))}
    runs = {Path(run).name: load_turn_records(resolve_path(Path(run) / "turns.csv")) for run in args.runs}
    started = datetime.now()
    report = render_weight_comparison(started, args.runs, args.variant,
                                      compare_weights(runs, variants["WICS"], variants["DFMS"], variants["OVS"]))
    print(report)

    report_path = resolve_path(Path(args.runs[0])).parent.parent / f"reports/{started:%Y-%m-%d_%H%M}_weights_{args.variant}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()