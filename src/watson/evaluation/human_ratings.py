"""Human rating sheet: checks the persona and narrative judges against people (thesis Section 1.4, Phase 3).

Replies are drawn from a run's turns.csv: the same number from each step, identical replies only once, in a
shuffled order with neutral ids, so raters cannot tell which configuration wrote a reply. The rating sheet holds
only what a rater needs; the answer key, kept apart, links each id back to its step and to the judges' scores.
The comparison reports how closely each rater (and the raters' average) follows the judges, the mean scores per
step, and how closely the raters agree with each other.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Sequence
from datetime import datetime
from itertools import combinations
from pathlib import Path
from statistics import mean

from pydantic import BaseModel

from watson.evaluation.middleware_regression import TurnRecord, history_before_each_turn
from watson.middleware.prompt_builder.builder import render_history

SHEET_FIELDS = ["item", "earlier_conversation", "visitor_message", "holmes_reply", "persona", "narrative", "comment"]
KEY_FIELDS = ["item", "step", "step_name", "dialogue_id", "turn", "kind", "judge_persona", "judge_narrative"]
SCORES = ("persona", "narrative")


class RatingItem(BaseModel):
    item: str
    step: int
    step_name: str
    dialogue_id: str
    turn: int
    kind: str
    earlier_conversation: str
    visitor_message: str
    holmes_reply: str
    judge_persona: int
    judge_narrative: int


def select_items(records: Sequence[TurnRecord], per_step: int, seed: int = 7, context_turns: int = 2) -> list[RatingItem]:
    """`per_step` generated replies from each step (fewer if a step has fewer), shuffled and given neutral ids.

    The fixed fallback line and identical replies (steps that share a reply) are left out, so raters do not
    score the same text twice.
    """
    rng = random.Random(seed)
    histories = history_before_each_turn(list(records))
    seen: set[str] = set()
    chosen: list[TurnRecord] = []
    for step in sorted({r.step for r in records}):
        candidates = []
        for r in sorted((r for r in records if r.step == step), key=lambda r: (r.dialogue_id, r.turn)):
            if r.source == "generator" and r.persona_score is not None and r.narrative_score is not None \
                    and r.reply not in seen:
                candidates.append(r)
                seen.add(r.reply)
        chosen += rng.sample(candidates, min(per_step, len(candidates)))
    rng.shuffle(chosen)
    return [
        RatingItem(
            item=f"R{i:02d}",
            step=r.step,
            step_name=r.step_name,
            dialogue_id=r.dialogue_id,
            turn=r.turn,
            kind=r.kind.value,
            earlier_conversation=render_history(histories[(r.step, r.dialogue_id, r.turn)][-context_turns:]),
            visitor_message=r.message,
            holmes_reply=r.reply,
            judge_persona=r.persona_score,
            judge_narrative=r.narrative_score,
        )
        for i, r in enumerate(chosen, start=1)
    ]


def write_sheet_and_key(items: Sequence[RatingItem], folder: Path) -> tuple[Path, Path]:
    """utf-8-sig so that Excel on Windows opens the files with the right characters."""
    folder.mkdir(parents=True, exist_ok=True)
    sheet, key = folder / "rating_sheet.csv", folder / "answer_key.csv"
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        writer.writerows({**item.model_dump(include=set(SHEET_FIELDS)), "persona": "", "narrative": "", "comment": ""}
                         for item in items)
    with key.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_FIELDS)
        writer.writeheader()
        writer.writerows(item.model_dump(include=set(KEY_FIELDS)) for item in items)
    return sheet, key


def read_key(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_ratings(path: Path, items: Sequence[str]) -> dict[str, dict[str, int]]:
    """A rater's filled-in sheet as {item: {"persona": n, "narrative": n}}; every item needs both scores, 1-10."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ratings, problems = {}, []
    for row in rows:
        item = row.get("item", "").strip()
        if item not in items:
            problems.append(f"unknown item {item!r}")
            continue
        scores = {}
        for score in SCORES:
            text = (row.get(score) or "").strip()
            if not text.isdigit() or not 1 <= int(text) <= 10:
                problems.append(f"{item} {score}: {text!r} is not a whole number from 1 to 10")
            else:
                scores[score] = int(text)
        ratings[item] = scores
    problems += [f"{item} is missing" for item in items if item not in ratings]
    if problems:
        raise ValueError(f"{path.name}: " + "; ".join(problems))
    return ratings


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1  # tied values share their average rank
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman's rank correlation; None when it is undefined (fewer than 3 pairs, or no variation)."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    return round(cov / var**0.5, 2) if var else None


def render_comparison(started: datetime, key: Sequence[dict[str, str]], raters: dict[str, dict[str, dict[str, int]]]) -> str:
    items = [row["item"] for row in key]
    judge = {score: [int(row[f"judge_{score}"]) for row in key] for score in SCORES}
    average = {score: [mean(r[item][score] for r in raters.values()) for item in items] for score in SCORES}

    def fmt(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}"

    lines = [
        f"# Human ratings compared with the judges: {started:%Y-%m-%d %H:%M}",
        "",
        f"{len(items)} replies, rated by {len(raters)} rater(s): {', '.join(sorted(raters))}.",
        "",
        "## Agreement with the judges",
        "",
        "Spearman's rank correlation between human scores and the judges' scores (1 = same order, 0 = unrelated).",
        "",
        "| Rater | Persona | Narrative |",
        "|---|---|---|",
    ]
    for name in sorted(raters):
        human = {score: [raters[name][item][score] for item in items] for score in SCORES}
        lines.append(f"| {name} | " + " | ".join(fmt(spearman(human[s], judge[s])) for s in SCORES) + " |")
    lines.append("| average of raters | " + " | ".join(fmt(spearman(average[s], judge[s])) for s in SCORES) + " |")

    lines += ["", "## Mean scores per step", "",
              "| Step | Name | Replies | Persona (humans) | Persona (judge) | Narrative (humans) | Narrative (judge) |",
              "|---" * 7 + "|"]
    for step in sorted({int(row["step"]) for row in key}):
        rows = [i for i, row in enumerate(key) if int(row["step"]) == step]
        cells = []
        for score in SCORES:
            cells += [fmt(mean(average[score][i] for i in rows)), fmt(mean(judge[score][i] for i in rows))]
        lines.append(f"| {step} | {key[rows[0]]['step_name']} | {len(rows)} | " + " | ".join(cells) + " |")

    if len(raters) > 1:
        lines += ["", "## Agreement between raters", "",
                  "Spearman's rank correlation, and the share of replies the two raters scored within 1 point.", "",
                  "| Raters | Persona | Persona within 1 | Narrative | Narrative within 1 |", "|---" * 5 + "|"]
        for a, b in combinations(sorted(raters), 2):
            cells = []
            for score in SCORES:
                xs = [raters[a][item][score] for item in items]
                ys = [raters[b][item][score] for item in items]
                within = sum(abs(x - y) <= 1 for x, y in zip(xs, ys)) / len(items)
                cells += [fmt(spearman(xs, ys)), f"{within:.0%}"]
            lines.append(f"| {a} and {b} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"