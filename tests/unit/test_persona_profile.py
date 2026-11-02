import json

from watson.common.persona import from_condensed, load_persona_profile
from watson.common.schemas import PersonaProfile
from watson.middleware.prompt_builder.builder import render_persona

GROUP = "data/persona_profiles/holmes_profile_condensed.json"
SAMPLE = "tests/fixtures/holmes_profile_sample.json"


def test_the_flat_sample_profile_loads_as_before():
    with open(SAMPLE, encoding="utf-8") as f:
        expected = PersonaProfile.model_validate(json.load(f))

    assert load_persona_profile(SAMPLE).model_dump(exclude={"created_at"}) == expected.model_dump(exclude={"created_at"})


def test_the_group_profile_fills_all_five_layers():
    profile = load_persona_profile(GROUP)

    assert profile.id == "holmes_profile_condensed"
    assert [len(getattr(profile, layer).traits) for layer in ("personality", "expression", "knowledge", "moral",
                                                              "belief")] == [8, 14, 14, 6, 11]


def test_the_group_profile_fits_the_generation_prompt():
    # About 1,900 tokens; with the rest of the prompt and a 400-token reply this stays inside Gemma's 4096.
    assert len(render_persona(load_persona_profile(GROUP))) < 8000


def test_entries_are_kept_word_for_word_except_the_sections_left_out():
    text = render_persona(load_persona_profile(GROUP))

    assert "- Openness (high): Curious about baffling puzzles; tests unconventional ideas" in text
    assert "- Address Watson with familiar professional warmth" in text  # conversational rules on their own
    assert '- Expertise, "Master": Observation, deduction' in text
    assert "Universalism (unsure)" in text  # values ranked low or unsure are kept
    assert "Reichenbach" not in text  # cases
    assert "Gloria Scott" not in text  # formative experiences
    assert "pipe-smoking" not in text  # non-verbal patterns


def test_each_entry_shape_becomes_a_trait_statement():
    data = {section: {} for section in ("personality_layer", "expression_layer", "moral_layer", "belief_layer",
                                         "canonical_knowledge")}
    data["personality_layer"]["big_five_traits"] = {"Openness": ["High", "Curious."]}
    data["expression_layer"]["conversational_rules"] = ["Speak formally."]
    data["belief_layer"]["core_worldview"] = "Evidence first."
    data["canonical_knowledge"]["cases"] = {"A Study in Scarlet": "Left out."}

    profile = from_condensed(data, profile_id="test")

    assert profile.personality.traits == ["Openness (high): Curious."]
    assert profile.expression.traits == ["Speak formally."]
    assert profile.belief.traits == ["Core worldview: Evidence first."]
    assert profile.knowledge.traits == []