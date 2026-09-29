import pytest
from pydantic import ValidationError

from watson.common.schemas import CriterionScore, DecisionType, RubricName, RubricScore
from watson.middleware.decision.decision import BandThresholds, decide, load_decision_thresholds
from watson.middleware.input_constraint.scoring import compute_wics, load_wics_weights

THRESHOLDS = load_decision_thresholds()
WEIGHTS = load_wics_weights()


def make_wics(oop: int, tr: int, hc: int, gip: int) -> RubricScore:
    scores = {"OOP": oop, "TR": tr, "HC": hc, "GIP": gip}
    return RubricScore(
        rubric=RubricName.WICS,
        criteria={code: CriterionScore(code=code, score=s) for code, s in scores.items()},
        weighted_total=compute_wics(scores, WEIGHTS),
    )


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ((9, 9, 9, 9), DecisionType.ACCEPT),
        ((8, 8, 8, 8), DecisionType.ACCEPT),  # WICS exactly 8.0, all criteria >= 7
        ((6, 6, 6, 6), DecisionType.REPHRASE),
        ((5, 5, 5, 5), DecisionType.REDIRECT),  # WICS 5.0 is in the redirect band
        ((3, 3, 3, 3), DecisionType.REJECT),  # WICS < 4.0
    ],
)
def test_table_5_2_rows(scores, expected):
    assert decide(make_wics(*scores), THRESHOLDS).decision == expected


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ((10, 10, 10, 6), DecisionType.REPHRASE),  # WICS 9.4, but one criterion is 6
        ((9, 9, 9, 4), DecisionType.REDIRECT),  # WICS 8.25, but one criterion is 4
        ((10, 10, 10, 2), DecisionType.REJECT),  # WICS 9.3, but a criterion is <= 2
    ],
)
def test_low_criterion_overrides_high_wics(scores, expected):
    assert decide(make_wics(*scores), THRESHOLDS).decision == expected


def test_gap_in_table_all_sevens_is_rephrase():
    # Table 5.2 matches no row here: WICS 7.0 is below Accept, and no criterion is in 5-6.
    assert decide(make_wics(7, 7, 7, 7), THRESHOLDS).decision == DecisionType.REPHRASE


def test_decision_explains_both_bands():
    result = decide(make_wics(10, 10, 10, 6), THRESHOLDS)

    assert result.reasons == [
        "WICS 9.4 is in the accept band",
        "lowest criterion GIP=6 is in the rephrase band",
    ]


def test_thresholds_must_be_in_descending_order():
    with pytest.raises(ValidationError):
        BandThresholds(accept=4.0, rephrase=6.0, redirect=8.0)