import pytest

from watson.common.schemas import (
    Constraint,
    ConstraintCategory,
    ConstraintStrengthText,
    CriterionScore,
    DialogueStage,
    RubricName,
    RubricScore,
)
from watson.common.rubric import weighted_score
from watson.middleware.dialogue_flow.constraint_selector import (
    ConstraintLibrary,
    load_selection_thresholds,
    select_constraints,
)
from watson.middleware.dialogue_flow.flow_evaluator import load_dfms_weights

LIBRARY = ConstraintLibrary.from_dir()
THRESHOLDS = load_selection_thresholds()
WEIGHTS = load_dfms_weights()


def make_dfms(cst: int, np: int, cc: int, tbc: int) -> RubricScore:
    scores = {"CST": cst, "NP": np, "CC": cc, "TBC": tbc}
    return RubricScore(
        rubric=RubricName.DFMS,
        criteria={code: CriterionScore(code=code, score=s) for code, s in scores.items()},
        weighted_total=weighted_score(scores, WEIGHTS),
    )


def selected(dfms: RubricScore, stage: DialogueStage) -> dict[str, int]:
    return {c.id: c.level for c in select_constraints(dfms, stage, LIBRARY, THRESHOLDS)}


# --- The repo's constraint library ------------------------------------------


def test_library_covers_all_six_categories():
    assert {c.category for c in LIBRARY.constraints} == set(ConstraintCategory)


def test_every_stage_has_exactly_one_dialogue_state_constraint():
    for stage in DialogueStage:
        assert len(LIBRARY.find(ConstraintCategory.DIALOGUE_STATE, stage)) == 1


def test_every_constraint_has_strength_levels_1_to_3():
    for constraint in LIBRARY.constraints:
        assert [s.level for s in constraint.strengths] == [1, 2, 3], constraint.id


def test_duplicate_constraint_ids_are_rejected():
    duplicate = Constraint(
        id="X", category=ConstraintCategory.PERSONA, strengths=[ConstraintStrengthText(level=1, text="t")]
    )
    with pytest.raises(ValueError, match="unique"):
        ConstraintLibrary([duplicate, duplicate])


# --- Selection rules ----------------------------------------------------------


def test_healthy_flow_adds_only_the_stage_constraint_and_the_always_on_ones():
    result = selected(make_dfms(9, 9, 9, 8), DialogueStage.EVIDENCE)

    assert result == {"DS-EVIDENCE": 1, "CC-NO-INVENTED-EVIDENCE": 2}


def test_weak_criteria_add_their_categories():
    result = selected(make_dfms(9, 4, 9, 9), DialogueStage.EVIDENCE)  # DFMS 7.5, NP weak

    assert result == {
        "DS-EVIDENCE": 1, "NP-ONE-STEP": 1, "NP-NO-PREMATURE-CONCLUSION": 1, "CC-NO-INVENTED-EVIDENCE": 2,
    }


def test_weak_state_tracking_firms_up_the_stage_constraint():
    assert selected(make_dfms(5, 9, 9, 9), DialogueStage.SUSPECTS)["DS-SUSPECTS"] == 2


def test_low_dfms_without_a_weak_criterion_uses_the_lowest():
    result = selected(make_dfms(8, 8, 7, 7), DialogueStage.EVIDENCE)  # DFMS 7.65, nothing below 7

    assert "CC-REFER-BACK" in result  # CC and TBC tie at 7; CC comes first


def test_very_low_dfms_applies_strength_level_2():
    result = selected(make_dfms(3, 4, 4, 5), DialogueStage.EVIDENCE)

    assert set(result.values()) == {2}
    assert "TB-STAY-ON-CASE" in result

def test_an_always_on_constraint_is_raised_to_its_level_when_its_category_is_weak():
    result = selected(make_dfms(9, 9, 3, 9), DialogueStage.EVIDENCE)  # DFMS 7.8, CC weak at level 1

    assert result["CC-REFER-BACK"] == 1
    assert result["CC-NO-INVENTED-EVIDENCE"] == 2


def test_an_always_on_constraint_never_lowers_a_level_the_dfms_asked_for():
    thresholds = THRESHOLDS.model_copy(update={"always": {"CC-NO-INVENTED-EVIDENCE": 1}})

    result = select_constraints(make_dfms(3, 4, 3, 5), DialogueStage.EVIDENCE, LIBRARY, thresholds)

    assert {c.id: c.level for c in result}["CC-NO-INVENTED-EVIDENCE"] == 2


def test_an_unknown_always_on_constraint_is_rejected():
    thresholds = THRESHOLDS.model_copy(update={"always": {"CC-MISSPELLED": 2}})

    with pytest.raises(ValueError, match="CC-MISSPELLED"):
        select_constraints(make_dfms(9, 9, 9, 9), DialogueStage.EVIDENCE, LIBRARY, thresholds)

def test_stage_limited_constraints_are_skipped_outside_their_stages():
    result = selected(make_dfms(9, 4, 9, 9), DialogueStage.DEDUCTION)

    assert "NP-NO-PREMATURE-CONCLUSION" not in result
    assert "NP-ONE-STEP" in result