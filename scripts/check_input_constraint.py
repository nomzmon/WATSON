"""Quick check of the Input Constraint Engine against an earlier regression run.

Re-scores every scripted message after each step's recorded conversation
history, with the current WICS prompt and decision rules, and reports how many
decisions are as expected and whether each message gets the same decision
after every history. Only the judge is called (no replies are generated), so it
takes a few minutes instead of an hour.
Usage (from the repo root):
    python scripts/check_input_constraint.py experiments/middleware_regression/runs/<run>/turns.csv
    python scripts/check_input_constraint.py <turns.csv> --steps 2 3
"""

from __future__ import annotations

import argparse
from datetime import datetime

from watson.common.config import get_settings, resolve_path
from watson.evaluation.input_check_stability import Check, TurnCheck, check_input_stability, render_stability_report
from watson.evaluation.middleware_regression import load_dialogues, load_experiment_config, load_turn_records
from watson.evaluation.regression_report import code_version
from watson.llm.llama_client import LlamaClient
from watson.middleware.decision.decision import load_decision_thresholds
from watson.middleware.input_constraint.engine import InputConstraintEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("turns", help="turns.csv from a regression run")
    parser.add_argument("--steps", type=int, nargs="+",
                        help="whose histories to use (default: every step in the file that had the input check)")
    parser.add_argument("--config", default="configs/experiments/middleware_regression.yaml")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    records = load_turn_records(args.turns)
    checked_steps = {r.step for r in records if r.wics_total is not None} or {r.step for r in records}
    steps = sorted(set(args.steps or checked_steps))
    started = datetime.now()
    messages = len({(r.dialogue_id, r.turn) for r in records})
    print(f"{messages} messages x {len(steps)} histories = {messages * len(steps)} judge calls.\n")

    engine = InputConstraintEngine.from_config(LlamaClient.from_settings(get_settings(), timeout=300))

    def show(turn: TurnCheck, check: Check) -> None:
        criteria = " ".join(f"{code} {score}" for code, score in check.criteria.items())
        expected = "" if not turn.expected else (" ok" if check.decision in turn.expected else " UNEXPECTED")
        print(f"  {turn.dialogue_id} turn {turn.turn} [{turn.kind.value}] after step {check.history_step}: "
              f"{check.decision.value}, WICS {check.wics:.2f} ({criteria}){expected}")

    results = check_input_stability(records, steps, engine, load_decision_thresholds(),
                                    load_dialogues(config.dialogues), config.expected_decisions, on_check=show)

    rated = [r for r in results if r.expected]
    print()
    for step in steps:
        print(f"Decisions as expected after step {step}'s history: {sum(r.as_expected(step) for r in rated)}/{len(rated)}")
    print(f"Same decision after every history: {sum(r.consistent for r in results)}/{len(results)} messages")

    report_path = resolve_path(f"experiments/middleware_regression/reports/{started:%Y-%m-%d_%H%M}_input_check.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_stability_report(started, code_version(), args.turns, steps, results), encoding="utf-8")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()