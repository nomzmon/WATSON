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


def test_from_config_applies_generator_yaml():
    generator = ResponseGenerator.from_config(Settings(generator_model="gemma:7b-instruct-q4_0"))

    assert generator.client.model_name == "gemma:7b-instruct-q4_0"
    assert generator.client.default_options == {"temperature": 0.7, "num_predict": 400, "num_ctx": 8192}
    assert generator.stop == ["Visitor:"]