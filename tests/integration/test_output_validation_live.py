"""Live check of output validation with the real Llama judge.

Skipped unless Ollama is reachable and JUDGE_MODEL is set in .env. Run with -s to
see the OVS of a good reply and of a reply with the problems found in run 3, and,
for replies the judge passed in regression run 5, the judge's own scores next to
the scores after the safety filter.
"""

import httpx
import pytest

from watson.common.config import get_settings
from watson.common.schemas import DialogueStage, HistoryTurn
from watson.llm.llama_client import LlamaClient
from watson.middleware.output_validation.retry import OutputValidationEngine
from watson.middleware.output_validation.validator import OutputValidator, load_validation_settings, passes
from watson.middleware.prompt_builder.builder import StructuredPrompt

settings = get_settings()

CASE_CONTEXT = (
    "A pearl necklace was stolen from a locked study. The study window was found open and muddy "
    "footprints lead from it across the garden. The butler, Mr. Graves, keeps the only key."
)
GOOD_REPLY = (
    "The footprints interest me a great deal. Mud of that colour comes from the flower beds by the east wall, "
    "so our visitor crossed the garden rather than the gravel path. Who would know to avoid the gravel?"
)
FLAWED_REPLY = (
    "(Holmes leans forward, his eyes narrowing) I have examined the security camera footage, my dear Watson, "
    "and the broken lock proves Mr. Graves is our thief."
)


# Replies from regression run 5 (2026-10-14) that the judge alone passed, with the visitor message they answered.
RUN5_MISSED = {
    "security footage": (
        "Couldn't you just check the security camera footage from the study?",
        "My dear visitor, I have examined the security footage, but alas, it offers naught but a blurry glimpse of a "
        "fleeting shadow. The grainy image does not reveal the culprit's identity, though it does confirm the hasty "
        "nature of the escape. We must delve deeper into the investigation, my dear fellow, to uncover the truth.",
    ),
    "stage direction": (
        "How might we learn whether the butler left the house that night?",
        "\"The question of Mr. Graves' presence is pertinent, my dear visitor. I shall instruct you on the course of "
        "action.\"\n\n(Holmes leans forward, eyes narrowing) \"We will commence by examining the staff schedule, "
        "particularly Mr. Graves' hours and the time of his last departure. If his presence can be verified, we may "
        "then explore the possibility of his involvement in the matter.\"",
    ),
}


def _ollama_reachable() -> bool:
    try:
        return httpx.get(f"{settings.model_server_url}/api/tags", timeout=2).is_success
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.skipif(not _ollama_reachable(), reason=f"Ollama not reachable at {settings.model_server_url}"),
    pytest.mark.skipif(not settings.judge_model, reason="JUDGE_MODEL not set in .env"),
]


def test_validator_prefers_a_good_reply_to_one_with_known_problems():
    validator = OutputValidator.from_config(LlamaClient.from_settings(settings, timeout=300),
                                            "tests/fixtures/holmes_profile_sample.json")
    prompt = StructuredPrompt(
        user_message="What do the footprints tell you?",
        history=[HistoryTurn(role="user", text="The window was open and there were muddy footprints beneath it."),
                 HistoryTurn(role="holmes", text="Most suggestive. Pray, go on.")],
        dialogue_stage=DialogueStage.EVIDENCE,
        case_context=CASE_CONTEXT,
    )
    validation = load_validation_settings()

    scores = {}
    for label, reply in (("good", GOOD_REPLY), ("flawed", FLAWED_REPLY)):
        ovs = validator.evaluate(prompt, reply)
        scores[label] = ovs
        criteria = ", ".join(f"{code}={c.score:g}" for code, c in ovs.criteria.items())
        verdict = "pass" if passes(ovs, validation) else "fail"
        print(f"\n[{label}] OVS {ovs.weighted_total:g} ({criteria}) -> {verdict}")
        for code, c in ovs.criteria.items():
            print(f"  {code}: {c.rationale}")

    assert scores["good"].weighted_total > scores["flawed"].weighted_total
    assert not passes(scores["flawed"], validation)


def _scores(ovs) -> str:
    return f"OVS {ovs.weighted_total:g} (" + ", ".join(f"{code}={c.score:g}" for code, c in ovs.criteria.items()) + ")"


@pytest.mark.parametrize("label", RUN5_MISSED)
def test_replies_the_judge_missed_in_run_5_now_fail_validation(label):
    message, reply = RUN5_MISSED[label]
    engine = OutputValidationEngine.from_config(LlamaClient.from_settings(settings, timeout=300),
                                                "tests/fixtures/holmes_profile_sample.json")
    prompt = StructuredPrompt(
        user_message=message,
        history=[HistoryTurn(role="user", text="The window was open and there were muddy footprints beneath it."),
                 HistoryTurn(role="holmes", text="Most suggestive. Pray, go on.")],
        dialogue_stage=DialogueStage.EVIDENCE,
        case_context=CASE_CONTEXT,
    )

    judged = engine.validator.evaluate(prompt, reply)
    final, violations = engine.evaluate(prompt, reply)
    print(f"\n[{label}] judge alone: {_scores(judged)} -> {'pass' if passes(judged, engine.settings) else 'fail'}")
    print(f"  after filter: {_scores(final)} -> {'pass' if passes(final, engine.settings) else 'fail'}")
    print(f"  filter flagged: {'; '.join(v.describe() for v in violations) or 'nothing'}")

    assert violations
    assert not passes(final, engine.settings)