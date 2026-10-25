import csv
from datetime import datetime

import pytest

from test_middleware_regression import run
from watson.evaluation.human_ratings import (
    SHEET_FIELDS,
    read_key,
    read_ratings,
    render_comparison,
    select_items,
    spearman,
    write_sheet_and_key,
)


def distinct_records():
    """Steps 1 and 2 of the fake regression run, each reply made unique so none is dropped as a duplicate."""
    return [r.model_copy(update={"reply": f"Reply {r.step}-{r.dialogue_id}-{r.turn}."}) for r in run(1) + run(2)]


def fill(sheet, path, scores):
    with sheet.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        for row, (persona, narrative) in zip(rows, scores):
            writer.writerow({**row, "persona": persona, "narrative": narrative})


# --- Choosing replies ----------------------------------------------------------------------


def test_the_same_number_of_replies_is_taken_from_each_step_in_shuffled_order():
    items = select_items(distinct_records(), per_step=3)

    assert [item.item for item in items] == ["R01", "R02", "R03", "R04", "R05", "R06"]
    assert sorted(item.step for item in items) == [1, 1, 1, 2, 2, 2]
    assert [i.holmes_reply for i in items] == [i.holmes_reply for i in select_items(distinct_records(), per_step=3)]


def test_identical_replies_and_the_fallback_line_are_left_out():
    records = [r.model_copy(update={"source": "fallback_message"}) if r.turn == 1 else r for r in run(1) + run(2)]

    items = select_items(records, per_step=10)

    assert [item.holmes_reply for item in items] == ["Elementary."]  # every generated reply is the same text


def test_each_item_shows_the_conversation_just_before_it():
    items = select_items(distinct_records(), per_step=20)

    later = next(item for item in items if item.turn == 3 and item.dialogue_id == "dialogue_01")
    assert later.earlier_conversation.startswith("Visitor: ")
    assert later.earlier_conversation.count("\n") == 1  # the last exchange only: one visitor line, one Holmes line


# --- The sheet, the key and the ratings ------------------------------------------------------


def test_the_sheet_is_blind_and_the_key_links_items_to_steps_and_judges(tmp_path):
    items = select_items(distinct_records(), per_step=2)

    sheet, key = write_sheet_and_key(items, tmp_path)

    with sheet.open(newline="", encoding="utf-8-sig") as f:
        assert next(csv.reader(f)) == SHEET_FIELDS  # no step, configuration or judge score
    assert [(row["item"], row["step"], row["judge_persona"]) for row in read_key(key)] == [
        (item.item, str(item.step), str(item.judge_persona)) for item in items
    ]


def test_ratings_are_read_and_checked(tmp_path):
    items = select_items(distinct_records(), per_step=2)
    sheet, _ = write_sheet_and_key(items, tmp_path)
    ids = [item.item for item in items]

    fill(sheet, tmp_path / "rated_ana.csv", [(8, 7)] * len(ids))
    assert read_ratings(tmp_path / "rated_ana.csv", ids)[ids[0]] == {"persona": 8, "narrative": 7}

    fill(sheet, tmp_path / "rated_ben.csv", [(11, 7), ("", 5)] + [(8, 7)] * (len(ids) - 3))
    with pytest.raises(ValueError) as error:
        read_ratings(tmp_path / "rated_ben.csv", ids)
    message = str(error.value)
    assert "'11' is not a whole number from 1 to 10" in message
    assert f"{ids[1]} persona" in message
    assert f"{ids[-1]} is missing" in message


# --- Comparison ----------------------------------------------------------------------------


@pytest.mark.parametrize(("xs", "ys", "expected"), [
    ([1, 2, 3, 4], [10, 20, 30, 40], 1.0),
    ([1, 2, 3, 4], [4, 3, 2, 1], -1.0),
    ([1, 2, 2, 3], [1, 2, 2, 3], 1.0),  # ties share their average rank
    ([5, 5, 5], [1, 2, 3], None),  # no variation
    ([1, 2], [1, 2], None),  # too few pairs
])
def test_spearman(xs, ys, expected):
    assert spearman(xs, ys) == expected


def test_comparison_reports_judge_agreement_step_means_and_rater_agreement(tmp_path):
    items = select_items(distinct_records(), per_step=3)
    _, key_path = write_sheet_and_key(items, tmp_path)
    key = read_key(key_path)
    ids = [item.item for item in items]
    raters = {
        "ana": {item: {"persona": 7, "narrative": 6 + i % 2} for i, item in enumerate(ids)},
        "ben": {item: {"persona": 8, "narrative": 6 + i % 2} for i, item in enumerate(ids)},
    }

    report = render_comparison(datetime(2026, 10, 25), key, raters)

    assert "6 replies, rated by 2 rater(s): ana, ben." in report
    assert "| ana | - | - |" in report  # the fake judge gives every reply the same scores, so no rank order
    assert "| 1 | persona prompt only | 3 | 7.50 | 7.00 |" in report
    assert "| ana and ben | - | 100% | 1.00 | 100% |" in report