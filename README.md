# WATSON — Curbing the Drift

Workflow Architecture for Tracking and Structuring Ongoing Narratives: an interaction-layer
middleware that reduces persona drift in a Sherlock Holmes role-playing chatbot.

Thesis: *Curbing the Drift: How Regulated Conversational Flows Shape LLM Persona Consistency*
De La Salle University, College of Computer Studies.

## Repository map
| Folder | Purpose |
|---|---|
| `docs/` | Thesis, architecture diagrams, ethics templates, meeting notes |
| `configs/` | Model settings, rubric weights/thresholds, 4 system configurations, experiments |
| `data/` | Canon corpus, preprocessing outputs, persona profiles, scenarios, labeled test inputs |
| `prompts/` | Extraction, judge, generation, and decision prompt templates |
| `src/watson/` | Persona construction, middleware components, LLM clients, pipeline, logging, evaluation, API |
| `frontend/` | Participant chat interface |
| `experiments/` | Run artifacts for framework validation, weight variance, participant study |
| `evaluation_instruments/` | Questionnaires and human annotation materials |
| `notebooks/`, `scripts/`, `tests/` | Analysis, CLI entry points, automated tests |
| `results/` | Final figures and tables for the manuscript |
| `models/`, `logs/` | Local weights and runtime logs (gitignored) |

Each folder has a `CONTENTS.txt` describing what belongs in it.

## Getting started
1. `cp .env.example .env` and fill in values
2. Set up a virtual environment and install dependencies (add `requirements.txt` / `pyproject.toml`)
3. Follow `scripts/CONTENTS.txt` for the build and run order
