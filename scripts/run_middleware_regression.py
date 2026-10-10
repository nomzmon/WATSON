"""Component-by-component middleware regression test.

Replays the scripted conversations through each step in
configs/experiments/middleware_regression.yaml, scores every generated Holmes
reply for persona and narrative consistency, and compares each step with the
one before it. Needs Ollama running with GENERATOR_MODEL and JUDGE_MODEL set in .env.
Usage (from the repo root):
    python scripts/run_middleware_regression.py              # every step
    python scripts/run_middleware_regression.py --steps 3 4  # only some steps
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime

from watson.common.config import get_settings, resolve_path
from watson.evaluation.middleware_regression import (
    FlowSummary,
    InputCheckSummary,
    StepRegression,
    StepSummary,
    TurnRecord,
    build_pipeline,
    find_regressions,
    load_dialogues,
    load_experiment_config,
    run_step,
    summarize_dialogue_flow,
    summarize_input_checks,
    summarize_step,
)
from watson.evaluation.narrative_consistency import NarrativeConsistencyJudge
from watson.evaluation.persona_consistency import PersonaConsistencyJudge
from watson.evaluation.regression_report import RunDetails, code_version, render_report
from watson.llm.llama_client import LlamaClient
from watson.middleware.generation.generator import ResponseGenerator

SECONDS_PER_MODEL_CALL = 15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/experiments/middleware_regression.yaml")
    parser.add_argument("--steps", type=int, nargs="+", help="step numbers to run (default: all)")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    numbers = args.steps or list(range(1, len(config.steps) + 1))
    if not set(numbers) <= set(range(1, len(config.steps) + 1)):
        parser.error(f"--steps must be between 1 and {len(config.steps)}")
    steps = {number: config.steps[number - 1] for number in sorted(set(numbers))}
    dialogues = load_dialogues(config.dialogues)
    turns = sum(len(d.turns) for d in dialogues)
    # Per turn: generation, persona and narrative judges, plus one judge call per enabled checking component.
    calls = sum(turns * (3 + step.modules.input_constraint + step.modules.dialogue_flow) for step in steps.values())
    started = datetime.now()
    out_dir = resolve_path(f"experiments/middleware_regression/runs/{started:%Y-%m-%d_%H%M}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Steps {', '.join(map(str, steps))} x {len(dialogues)} conversations ({turns} turns each step), "
          f"about {calls} model calls: roughly {calls * SECONDS_PER_MODEL_CALL // 60} minutes.")
    print(f"Turns are saved as they finish to {out_dir / 'turns.csv'}\n")

    settings = get_settings()
    details = RunDetails(
        started=started,
        code_version=code_version(),
        generator_model=settings.generator_model or "not set",
        judge_model=settings.judge_model or "not set",
    )
    judge_client = LlamaClient.from_settings(settings, timeout=300)
    generator = ResponseGenerator.from_config(settings)
    persona_judge = PersonaConsistencyJudge.from_config(judge_client, config.profile)
    narrative_judge = NarrativeConsistencyJudge.from_config(judge_client)

    summaries, all_records = [], []
    with (out_dir / "turns.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TurnRecord.model_fields))
        writer.writeheader()

        def record(r: TurnRecord) -> None:
            writer.writerow(r.model_dump(mode="json"))
            f.flush()
            parts = []
            if r.wics_total is not None:
                criteria = " ".join(f"{code} {score}" for code, score in r.wics_scores().items())
                parts.append(f"{r.decision.value}, WICS {r.wics_total:.2f} ({criteria})")
            if r.dfms_total is not None:
                parts.append(f"{r.stage.value}, DFMS {r.dfms_total:.2f}")
            if r.persona_score is not None:
                parts.append(f"persona {r.persona_score}/10, narrative {r.narrative_score}/10")
            else:
                parts.append(r.source.replace("_", " "))
            print(f"  {r.dialogue_id} turn {r.turn} [{r.kind.value}]: {' -> '.join(parts)}")

        for index, step in steps.items():
            print(f"Step {index}: {step.name}")
            pipeline = build_pipeline(step, config.profile, generator, judge_client)
            records = run_step(index, step, pipeline, persona_judge, dialogues, config.seed, on_record=record,
                               narrative_judge=narrative_judge)
            summaries.append(summarize_step(index, step.name, records))
            all_records += records

    regressions = find_regressions(summaries, config.regression_tolerance)
    input_checks = summarize_input_checks(all_records, config.expected_decisions)
    flows = summarize_dialogue_flow(all_records)
    print_report(summaries, regressions, config.regression_tolerance)
    print_input_checks(input_checks)
    print_flows(flows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {"steps": [s.model_dump() for s in summaries],
             "input_checks": [s.model_dump(mode="json") for s in input_checks],
             "dialogue_flow": [s.model_dump(mode="json") for s in flows]},
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path = resolve_path(f"experiments/middleware_regression/reports/{started:%Y-%m-%d_%H%M}_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(details, config, dialogues, summaries, regressions, all_records, input_checks, flows),
        encoding="utf-8",
    )
    print(f"\nEvery turn, reply and the judge's reasoning: {out_dir / 'turns.csv'}")
    print(f"Report for your documentation (commit this): {report_path}")


def print_report(summaries: list[StepSummary], regressions: list[StepRegression], tolerance: float) -> None:
    def score(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "-"

    print(f"\n{'Step':<44}{'Persona':>9}{'Persona':>9}{'Narrat.':>9}{'Repeti-':>9}{'Drift':>9}{'Normal':>9}"
          f"{'Avg time':>10}")
    print(f"{'':<44}{'normal':>9}{'drift':>9}{'normal':>9}{'tion':>9}{'caught':>9}{'blocked':>9}{'per turn':>10}")
    for s in summaries:
        print(f"{s.step:<2} {s.name:<41}{score(s.persona_normal):>9}{score(s.persona_drift):>9}"
              f"{score(s.narrative_normal):>9}{score(s.repetition):>9}"
              f"{f'{s.drift_caught}/{s.drift_total}':>9}{f'{s.normal_blocked}/{s.normal_total}':>9}"
              f"{s.mean_latency_ms / 1000:>9.1f}s")

    print("\nPersona normal = mean persona score (1-10) of Holmes' replies to normal turns.")
    print("Persona drift  = the same for drift-inducing turns that still reached Holmes.")
    print("Narrat. normal = mean narrative score (1-10): true to the case, moving forward, naturally paced.")
    print("Repetition     = share (0-1) of each reply's three-word sequences Holmes already used earlier.")
    print("Drift caught   = drift-inducing turns the middleware redirected or rejected.")
    print("Normal blocked = normal turns the middleware wrongly redirected or rejected.")

    print(f"\nRegressions (persona normal falling more than {tolerance:g} from the previous step):")
    if not regressions:
        print("  none")
    for r in regressions:
        print(f"  step {r.step} dropped {r.drop:.2f}")


def print_input_checks(summaries: list[InputCheckSummary]) -> None:
    if not summaries:
        return
    print("\nInput check scores (mean per turn kind; decisions accept/rephrase/redirect/reject):")
    print(f"{'Step':<6}{'Turn kind':<18}{'Turns':>6}{'WICS':>7}{'OOP':>6}{'TR':>6}{'HC':>6}{'GIP':>6}"
          f"   Decisions   As expected")
    for s in summaries:
        decisions = "/".join(str(count) for count in s.decisions.values())
        as_expected = f"{s.as_expected}/{s.turns}" if s.as_expected is not None else "-"
        print(f"{s.step:<6}{s.kind.value:<18}{s.turns:>6}{s.wics:>7.2f}"
              + "".join(f"{score:>6.1f}" for score in s.criteria.values()) + f"   {decisions:<12}{as_expected}")
    for step in dict.fromkeys(s.step for s in summaries):
        rated = [s for s in summaries if s.step == step and s.as_expected is not None]
        if rated:
            print(f"Decisions as expected in step {step}: "
                  f"{sum(s.as_expected for s in rated)}/{sum(s.turns for s in rated)}")


def print_flows(summaries: list[FlowSummary]) -> None:
    if not summaries:
        return
    print("\nDialogue flow (mean DFMS before each reply, guided turns, final stage per conversation):")
    print(f"{'Step':<6}{'Turns':>6}{'DFMS':>7}{'CST':>6}{'NP':>6}{'CC':>6}{'TBC':>6}{'Guided':>9}   Final stages")
    for s in summaries:
        stages = ", ".join(f"{dialogue}: {stage.value}" for dialogue, stage in s.final_stages.items())
        print(f"{s.step:<6}{s.turns:>6}{s.dfms:>7.2f}" + "".join(f"{v:>6.1f}" for v in s.criteria.values())
              + f"{f'{s.guided_turns}/{s.turns}':>9}   {stages}")


if __name__ == "__main__":
    main()