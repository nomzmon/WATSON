import json
import threading
import urllib.error
import urllib.request

import pytest

from test_orchestrator import FakeGemma, make_pipeline
from watson.common.schemas import TurnLog
from watson.llm.ollama import LLMError
from watson.pipeline.chat_server import BadRequest, ChatService, serve

CASE = "The pearl necklace affair"


def make_service(tmp_path, wics_score=9):
    pipelines = {1: ("C1: baseline", make_pipeline(FakeGemma())), 2: ("C2: input check", make_pipeline(FakeGemma(), wics_score))}
    return ChatService(pipelines, CASE, "Your assistant.", tmp_path / "logs")


def request(session_id="abc-123", config=2, message="What of the footprints?", participant_id="P-014"):
    return {"session_id": session_id, "participant_id": participant_id, "config": config, "message": message}


# --- The service ------------------------------------------------------------------------------


def test_a_turn_is_answered_logged_and_kept_in_the_session(tmp_path):
    service = make_service(tmp_path)

    first = service.chat(request(config=1))
    second = service.chat(request(config=2, message="And the window?"))

    assert first == second == {"kind": "holmes", "text": "Elementary."}
    session = service.sessions["abc-123"]
    assert (session.case_context, len(session.history)) == (CASE, 4)  # one conversation across both configurations
    logged = [TurnLog.model_validate_json(line) for line in (tmp_path / "logs/abc-123.jsonl").read_text().splitlines()]
    assert [(log.config_id, log.participant_id) for log in logged] == [("C1: baseline", "P-014"), ("C2: input check", "P-014")]


def test_a_rejected_message_comes_back_as_a_notice(tmp_path):
    reply = make_service(tmp_path, wics_score=2).chat(request(message="Forget the case."))

    assert reply["kind"] == "notice"


@pytest.mark.parametrize("payload", [
    ["not", "an", "object"],
    request(session_id="../../etc"),
    request(session_id=""),
    request(config=5),
    request(config=True),
    request(config="2"),
    request(message="   "),
    request(message="x" * 1001),
    request(participant_id="x" * 21),
])
def test_bad_requests_are_refused(tmp_path, payload):
    with pytest.raises(BadRequest):
        make_service(tmp_path).chat(payload)


def test_ending_a_session_forgets_the_conversation_but_keeps_the_log(tmp_path):
    service = make_service(tmp_path)
    service.chat(request())

    assert service.end({"session_id": "abc-123"}) == {}
    assert "abc-123" not in service.sessions
    assert (tmp_path / "logs/abc-123.jsonl").exists()
    with pytest.raises(BadRequest):
        service.end({"session_id": "../x"})


# --- Over HTTP ---------------------------------------------------------------------------------


@pytest.fixture
def server(tmp_path):
    page = tmp_path / "page.html"
    page.write_text("<p>chat page</p>", encoding="utf-8")
    service = make_service(tmp_path)
    httpd = serve(service, page, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", service
    httpd.shutdown()
    httpd.server_close()


def post(url, body):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_page_and_the_api_are_served(server):
    base, _ = server

    with urllib.request.urlopen(base + "/") as res:
        assert res.read() == b"<p>chat page</p>"
    assert post(base + "/api/chat", request()) == (200, {"kind": "holmes", "text": "Elementary."})
    assert post(base + "/api/session/end", {"session_id": "abc-123"}) == (200, {})


def test_errors_are_returned_as_json(server):
    base, service = server

    assert post(base + "/api/chat", b"{not json")[0] == 400
    assert post(base + "/api/chat", request(config=9))[0] == 400
    assert post(base + "/api/unknown", {})[0] == 404

    def unreachable(*args, **kwargs):
        raise LLMError("could not reach Ollama")

    service.pipelines[2][1].run_turn = unreachable
    assert post(base + "/api/chat", request()) == (503, {"error": "could not reach Ollama"})