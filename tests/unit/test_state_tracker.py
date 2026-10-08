from watson.common.schemas import DialogueStage
from watson.middleware.dialogue_flow.state_tracker import next_stage

INTRODUCTION, EVIDENCE, SUSPECTS, DEDUCTION, RESOLUTION = DialogueStage


def test_advances_one_stage_when_the_judge_says_so():
    assert next_stage(EVIDENCE, SUSPECTS) == SUSPECTS


def test_never_skips_ahead_more_than_one_stage():
    assert next_stage(INTRODUCTION, RESOLUTION) == EVIDENCE


def test_never_moves_back_to_a_completed_stage():
    assert next_stage(DEDUCTION, EVIDENCE) == DEDUCTION


def test_stays_when_the_judge_proposes_the_same_stage():
    assert next_stage(SUSPECTS, SUSPECTS) == SUSPECTS


def test_resolution_is_final():
    assert next_stage(RESOLUTION, RESOLUTION) == RESOLUTION