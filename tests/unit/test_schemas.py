from watson.common.schemas import (
    ComponentLatency,
    Constraint,
    ConstraintCategory,
    ConstraintStrengthText,
    CriterionScore,
    DecisionResult,
    DecisionType,
    PersonaLayer,
    PersonaProfile,
    RubricName,
    RubricScore,
    TurnLog,
)


def make_persona_profile() -> PersonaProfile:
    layer = PersonaLayer(traits=["observant", "logical"])
    return PersonaProfile(
        personality=layer,
        expression=layer,
        knowledge=layer,
        moral=layer,
        belief=layer,
    )


def test_persona_profile_defaults_to_draft():
    profile = make_persona_profile()
    assert profile.status == "draft"
    assert profile.id.startswith("persona_")


def test_rubric_score_holds_criteria_by_code():
    score = RubricScore(
        rubric=RubricName.WICS,
        criteria={"OOP": CriterionScore(code="OOP", score=4.5)},
        weighted_total=4.5,
        passed=True,
    )
    assert score.criteria["OOP"].score == 4.5


def test_constraint_text_for_level():
    constraint = Constraint(
        id="c1",
        category=ConstraintCategory.PERSONA,
        strengths=[
            ConstraintStrengthText(level=1, text="mild reminder"),
            ConstraintStrengthText(level=2, text="firm reminder"),
        ],
    )
    assert constraint.text_for(2) == "firm reminder"
    assert constraint.max_level == 2


def test_turn_log_total_latency_sums_components():
    log = TurnLog(
        session_id="s1",
        config_id="c4_watson_full",
        user_input="Who is the suspect?",
        final_response="Elementary, my dear fellow.",
        input_decision=DecisionResult(decision=DecisionType.ACCEPT),
    )
    log.latencies = [
        ComponentLatency(component="input_constraint", ms=10.0),
        ComponentLatency(component="generation", ms=250.0),
    ]
    assert log.total_latency_ms == 260.0