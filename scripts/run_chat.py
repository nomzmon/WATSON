"""Chat with Holmes in the terminal through one of an experiment's configurations.

The configuration is not shown during the chat, so a participant cannot tell which one they are using. Every
turn is logged to experiments/<experiment>/chat_logs/<date>_<session>.jsonl. Needs Ollama running with
GENERATOR_MODEL and JUDGE_MODEL set in .env. Send an empty message to finish.
Usage (from the repo root):
    python scripts/run_chat.py --step 4                       # C4, the complete framework
    python scripts/run_chat.py --step 1 --participant P07     # C1, logged under participant P07
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from watson.common.config import get_settings, resolve_path
from watson.evaluation.middleware_regression import build_pipeline, load_dialogues, load_experiment_config
from watson.llm.llama_client import LlamaClient
from watson.middleware.generation.generator import ResponseGenerator
from watson.pipeline.chat import chat
from watson.pipeline.session import ConversationSession


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/experiments/framework_validation.yaml")
    parser.add_argument("--step", type=int, required=True, help="which configuration (step number) to use")
    parser.add_argument("--participant", help="participant id to record in the log")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if not 1 <= args.step <= len(config.steps):
        parser.error(f"--step must be between 1 and {len(config.steps)}")
    step = config.steps[args.step - 1]
    case = load_dialogues(config.dialogues)[0]  # every scripted dialogue uses the same case and visitor

    settings = get_settings()
    pipeline = build_pipeline(step, config.profile, ResponseGenerator.from_config(settings),
                              LlamaClient.from_settings(settings, timeout=300))
    session = ConversationSession(case_context=case.case_context, visitor=case.visitor)
    log_path = resolve_path(f"experiments/{Path(args.config).stem}/chat_logs/"
                            f"{datetime.now():%Y-%m-%d_%H%M}_{session.session_id}.jsonl")

    print("You are speaking with Sherlock Holmes. Send an empty message to finish.\n")
    turns = chat(pipeline, session, step.name, log_path, read=lambda: input("You: "),
                 write=lambda reply: print(f"\nHolmes: {reply}\n"), participant_id=args.participant)
    print(f"{turns} turns logged to {log_path}")


if __name__ == "__main__":
    main()