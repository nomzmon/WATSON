"""Backend for the participant chat page in frontend/ (thesis Section 1.4, Phase 4).

Implements the contract the page expects:
    POST /api/chat         {session_id, participant_id (may be null), config (1-4), message}
                           -> {kind: "holmes" | "notice", text}
    POST /api/session/end  {session_id} -> {}
Each browser session keeps one conversation; the configuration is chosen per message (the page lets it change),
and every turn is logged with the configuration that answered it. Requests are handled one at a time, since the
models share one GPU.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from watson.llm.ollama import LLMError
from watson.pipeline.chat import log_turn, to_turn_log
from watson.pipeline.orchestrator import Pipeline
from watson.pipeline.session import ConversationSession

_SESSION_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")  # also used in the log file name
MAX_PARTICIPANT_ID = 20  # the page's input limit; the id only goes inside the log
MAX_MESSAGE = 1000  # the page's textarea limit
MAX_BODY = 10_000


class BadRequest(ValueError):
    pass


class ChatService:
    def __init__(self, pipelines: Mapping[int, tuple[str, Pipeline]], case_context: str, visitor: str | None,
                 log_dir: Path) -> None:
        """`pipelines` maps the page's configuration number to (configuration name, pipeline)."""
        self.pipelines = dict(pipelines)
        self.case_context = case_context
        self.visitor = visitor
        self.log_dir = log_dir
        self.sessions: dict[str, ConversationSession] = {}

    def chat(self, payload: object) -> dict[str, str]:
        if not isinstance(payload, dict):
            raise BadRequest("the body must be a JSON object")
        session_id = payload.get("session_id")
        participant_id = payload.get("participant_id") or None
        config = payload.get("config")
        message = payload.get("message")
        if not isinstance(session_id, str) or not _SESSION_ID.match(session_id):
            raise BadRequest("session_id must be 1-64 letters, digits or hyphens")
        if participant_id is not None and (not isinstance(participant_id, str) or len(participant_id) > MAX_PARTICIPANT_ID):
            raise BadRequest(f"participant_id must be text of at most {MAX_PARTICIPANT_ID} characters")
        if not isinstance(config, int) or isinstance(config, bool) or config not in self.pipelines:
            raise BadRequest(f"config must be one of {sorted(self.pipelines)}")
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE:
            raise BadRequest(f"message must be 1-{MAX_MESSAGE} characters")

        session = self.sessions.setdefault(session_id, ConversationSession(
            session_id=session_id, case_context=self.case_context, visitor=self.visitor))
        name, pipeline = self.pipelines[config]
        result = pipeline.run_turn(session, message.strip())
        log_turn(self.log_dir / f"{session_id}.jsonl", to_turn_log(result, session, name, participant_id))
        return {"kind": "notice" if result.source == "reject_message" else "holmes", "text": result.reply}

    def end(self, payload: object) -> dict[str, str]:
        session_id = payload.get("session_id") if isinstance(payload, dict) else None
        if not isinstance(session_id, str) or not _SESSION_ID.match(session_id):
            raise BadRequest("session_id must be 1-64 letters, digits or hyphens")
        self.sessions.pop(session_id, None)  # the log stays; only the conversation in memory is dropped
        return {}


def make_handler(service: ChatService, page: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in ("/", "/index.html"):
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            body = page.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            routes = {"/api/chat": service.chat, "/api/session/end": service.end}
            if self.path not in routes:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise BadRequest(f"the body must be 1-{MAX_BODY} bytes")
                payload = json.loads(self.rfile.read(length))
                self._send(HTTPStatus.OK, routes[self.path](payload))
            except LLMError as error:  # Ollama is down or timed out; the page asks the visitor to send again
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
            except (BadRequest, ValueError) as error:  # json.JSONDecodeError is a ValueError
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def _send(self, status: HTTPStatus, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(service: ChatService, page: Path, host: str = "127.0.0.1", port: int = 8000) -> HTTPServer:
    return HTTPServer((host, port), make_handler(service, page))