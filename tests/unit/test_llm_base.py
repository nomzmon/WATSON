from watson.llm.base import LLMClient


class EchoClient(LLMClient):
    model_name = "echo-test"

    def _generate(self, prompt: str, **kwargs) -> str:
        return prompt.upper()


def test_generate_wraps_text_and_measures_latency():
    response = EchoClient().generate("hello")

    assert response.text == "HELLO"
    assert response.model == "echo-test"
    assert response.latency_ms >= 0


def test_generate_structured_validates_json_by_default():
    from pydantic import BaseModel

    class Greeting(BaseModel):
        text: str

    class JsonClient(LLMClient):
        model_name = "json-test"

        def _generate(self, prompt: str, **kwargs) -> str:
            return '{"text": "hi"}'

    result = JsonClient().generate_structured("prompt", Greeting)

    assert result.text == "hi"