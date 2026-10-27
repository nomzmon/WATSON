"""Serve the participant chat page with the WATSON configurations behind it.

Open http://127.0.0.1:8000 in a browser on the same computer. Config 1-4 on the page are C1-C4 of the
experiment config; every turn is logged to experiments/<experiment>/chat_logs/web/<session>.jsonl. Needs Ollama
running with GENERATOR_MODEL and JUDGE_MODEL set in .env. Stop the server with Ctrl+C.
Usage (from the repo root):
    python scripts/run_chat_server.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from watson.common.config import get_settings, resolve_path
from watson.evaluation.middleware_regression import build_pipeline, load_dialogues, load_experiment_config
from watson.llm.llama_client import LlamaClient
from watson.middleware.generation.generator import ResponseGenerator
from watson.pipeline.chat_server import ChatService, serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/experiments/framework_validation.yaml")
    parser.add_argument("--page", default="frontend/index(1).html")
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only (default)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    case = load_dialogues(config.dialogues)[0]  # every scripted dialogue uses the same case and visitor
    settings = get_settings()
    generator = ResponseGenerator.from_config(settings)
    judge = LlamaClient.from_settings(settings, timeout=300)
    pipelines = {number: (step.name, build_pipeline(step, config.profile, generator, judge))
                 for number, step in enumerate(config.steps, start=1)}
    log_dir = resolve_path(f"experiments/{Path(args.config).stem}/chat_logs/web")
    service = ChatService(pipelines, case.case_context, case.visitor, log_dir)

    server = serve(service, resolve_path(args.page), args.host, args.port)
    print(f"Chat page: http://{args.host}:{args.port}  (logs: {log_dir})  Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()