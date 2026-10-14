"""Live check of the Output Validation judge with the real Llama model.

Skipped unless Ollama is reachable and JUDGE_MODEL is set in .env. Run with -s to
see the OVS of a good reply and of a reply with the problems found in run 3.
"""

import httpx
import pytest

from watson.common.config import get_settings
from watson.common.schemas import DialogueStage, HistoryTurn
from watson.llm.llama_client import LlamaClient
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