"""Load a persona profile in either of the two formats the project uses.

- The flat format of PersonaProfile: five layers, each a list of trait statements
  (tests/fixtures/holmes_profile_sample.json, the synthetic stand-in).
- The group's condensed layered format (data/persona_profiles/holmes_profile_condensed.json):
  personality_layer, expression_layer, moral_layer, belief_layer and
  canonical_knowledge, whose entries are text, lists of text, or
  [level, description] pairs.

Every entry of the condensed format becomes one trait statement, word for word,
except the sections in LEFT_OUT: Gemma's 4096-token context leaves room for about
8,000 characters of persona, and the whole profile is about 9,700. The canonical
cases and formative experiences are background the role-play rarely needs, and
the non-verbal patterns invite described actions, which replies must not contain.
The file itself is never changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from watson.common.config import load_text
from watson.common.schemas import PersonaLayer, PersonaProfile

LAYERS = {
    "personality_layer": "personality",
    "expression_layer": "expression",
    "moral_layer": "moral",
    "belief_layer": "belief",
    "canonical_knowledge": "knowledge",
}
LEFT_OUT = {"cases", "formative_experiences", "non_verbal_expression_patterns"}


def load_persona_profile(path: str | Path) -> PersonaProfile:
    data = json.loads(load_text(path))
    if "personality_layer" in data:
        return from_condensed(data, profile_id=Path(path).stem)
    return PersonaProfile.model_validate(data)


def from_condensed(data: dict[str, Any], profile_id: str) -> PersonaProfile:
    return PersonaProfile(
        id=profile_id,
        version="condensed",
        **{layer: PersonaLayer(traits=[line for key, value in data[section].items() if key not in LEFT_OUT
                                       for line in _statements(key, value)])
           for section, layer in LAYERS.items()},
    )


def _statements(key: str, value: Any) -> list[str]:
    label = key.replace("_", " ").capitalize()
    if isinstance(value, str):
        return [f"{label}: {value}"]
    if isinstance(value, list):  # rules and steps read on their own
        return [str(item) for item in value]
    lines = []
    for name, entry in value.items():
        if isinstance(entry, list):  # [level, description], e.g. "Openness": ["High", "..."]
            lines.append(f"{name} ({str(entry[0]).lower()}): {entry[1]}")
        else:  # e.g. "Master": "Observation, ..." under expertise
            lines.append(f'{label}, "{name}": {entry}')
    return lines