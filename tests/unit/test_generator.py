import pytest

from watson.common.config import Settings
from watson.llm.base import LLMClient
from watson.middleware.generation.generator import ResponseGenerator, clean_response


class FakeGemma(LLMClient):
    model_name = "fake-gemma"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def _generate(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return self.reply


def test_generate_sends_prompt_with_stop_sequences():
    client = FakeGemma("Elementary.")
    generator = ResponseGenerator(client, stop=["Visitor:"])

    response = generator.generate("the structured prompt")

    assert response.text == "Elementary."
    assert client.calls == [{"prompt": "the structured prompt", "stop": ["Visitor:"]}]
    

def test_generate_passes_extra_options_such_as_seed():
    client = FakeGemma("Elementary.")

    ResponseGenerator(client, stop=["Visitor:"]).generate("prompt", seed=42)

    assert client.calls == [{"prompt": "prompt", "stop": ["Visitor:"], "seed": 42}]


def test_generate_strips_copied_name_label():
    generator = ResponseGenerator(FakeGemma("  Holmes: The butler, of course.  "))

    assert generator.generate("x").text == "The butler, of course."
    
def test_without_clean_up_the_raw_reply_is_returned():
    raw = "Holmes: (He leans forward) The butler, of course."

    reply = ResponseGenerator(FakeGemma(raw), clean=False).generate("x")

    assert (reply.text, reply.removed) == (raw, [])

@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("Sherlock Holmes: Quite so.", "Quite so."),
        ("holmes : Quite so.", "Quite so."),
        ("Quite so, Holmes: indeed.", "Quite so, Holmes: indeed."),  # only a leading label is removed
        ('"The footprints tell us much."', "The footprints tell us much."),
        ("\u201cThe footprints tell us much.\u201d", "The footprints tell us much."),
        ('Holmes: "Quite so."', "Quite so."),
        ('"Indeed," said I. "Most curious."', '"Indeed," said I. "Most curious."'),  # separate quotes kept
        ('He called it "elementary".', 'He called it "elementary".'),
    ],
)
def test_clean_response(raw, cleaned):
    assert clean_response(raw) == cleaned

@pytest.mark.parametrize(
    ("raw", "cleaned", "removed"),
    [
        # Replies from regression runs 5 and 6 that failed validation only because of the stage direction.
        ('"The question is pertinent."\n\n(Holmes leans forward, eyes narrowing) "We will examine the staff."',
         "The question is pertinent.\n\nWe will examine the staff.", ["(Holmes leans forward, eyes narrowing)"]),
        ('"Good evening, sir. Do tell me precisely what happened."\n\n(Holmes leans forward, his eyes narrowing)',
         "Good evening, sir. Do tell me precisely what happened.", ["(Holmes leans forward, his eyes narrowing)"]),
        ("*leans back in his chair* The mud is telling.", "The mud is telling.", ["*leans back in his chair*"]),
        ("[He lights his pipe.] Go on.", "Go on.", ["[He lights his pipe.]"]),
        ("The mud is telling (as I suspected), and the window was forced.",
         "The mud is telling (as I suspected), and the window was forced.", []),  # Holmes' own aside stays
        ("You must *see*, not merely look.", "You must *see*, not merely look.", []),  # one-word emphasis stays
        ("(Holmes leans forward)", "(Holmes leans forward)", []),  # nothing would be left, so validation decides
    ],
)
def test_generate_cuts_stage_directions(raw, cleaned, removed):
    reply = ResponseGenerator(FakeGemma(raw)).generate("x")

    assert (reply.text, reply.removed) == (cleaned, removed)


def test_a_single_unclosed_opening_quote_is_dropped():
    assert clean_response('"Indeed, the gardener\'s dismissal raises suspicion.') == (
        "Indeed, the gardener's dismissal raises suspicion."
    )


def test_from_config_applies_generator_yaml():
    generator = ResponseGenerator.from_config(Settings(generator_model="gemma:7b-instruct-q4_0"))

    assert generator.client.model_name == "gemma:7b-instruct-q4_0"
    assert generator.client.default_options == {"temperature": 0.7, "num_predict": 400, "num_ctx": 4096}
    assert generator.stop == ["Visitor:"]