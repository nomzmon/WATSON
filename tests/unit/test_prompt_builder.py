from watson.common.config import load_text
from watson.common.schemas import DialogueStage, HistoryTurn, PersonaLayer, PersonaProfile
from watson.middleware.prompt_builder.builder import PromptBuilder, render_persona

FIXTURE_PROFILE = "tests/fixtures/holmes_profile_sample.json"


def make_builder(template_path: str = "prompts/generation/structured_prompt.txt", **kwargs) -> PromptBuilder:
    return PromptBuilder.from_config(profile_path=FIXTURE_PROFILE, template_path=template_path, **kwargs)


def test_prompt_contains_every_section_from_section_5_1_5():
    builder = make_builder()
    history = [
        HistoryTurn(role="user", text="A necklace was stolen."),
        HistoryTurn(role="holmes", text="From a locked room, I presume."),
    ]
    prompt = builder.build(
        "Who had the key?",
        history=history,
        dialogue_stage=DialogueStage.SUSPECTS,
        case_context="The Blue Necklace affair",
    )
    prompt.constraints.append("Do not reveal the culprit yet.")

    text = builder.render(prompt)

    assert "Relentlessly analytical" in text  # persona profile
    assert "Who had the key?" in text  # validated user message
    assert "Visitor: A necklace was stolen.\nHolmes: From a locked room, I presume." in text  # history
    assert "Current stage of the investigation: suspects" in text  # dialogue state
    assert "The Blue Necklace affair" in text  # case context
    assert "- Do not reveal the culprit yet." in text  # constraint from the Dialogue Flow Manager


def test_defaults_when_context_is_missing():
    builder = make_builder()

    text = builder.render(builder.build("Good evening, Mr. Holmes."))

    assert "(This is the start of the conversation.)" in text
    assert "Current stage of the investigation: not yet tracked" in text
    assert "No case has been presented yet." in text
    assert "None for this reply." in text


def test_only_recent_history_is_kept():
    builder = make_builder(history_turns=2)
    history = [HistoryTurn(role="user", text=f"turn {i}") for i in range(5)]

    prompt = builder.build("next", history=history)

    assert [t.text for t in prompt.history] == ["turn 3", "turn 4"]


def test_baseline_template_uses_persona_and_history_only():
    builder = make_builder("prompts/generation/baseline_persona.txt")
    prompt = builder.build("Who had the key?", dialogue_stage=DialogueStage.SUSPECTS, case_context="secret case")
    prompt.constraints.append("hidden constraint")

    text = builder.render(prompt)

    assert "Relentlessly analytical" in text
    assert "Who had the key?" in text
    assert "secret case" not in text
    assert "hidden constraint" not in text


def test_braces_in_user_message_do_not_break_rendering():
    builder = make_builder()

    assert "What is {this}?" in builder.render(builder.build("What is {this}?"))


def test_persona_layers_are_rendered_under_headings_and_empty_layers_skipped():
    profile = PersonaProfile(
        personality=PersonaLayer(traits=["observant"]),
        expression=PersonaLayer(traits=["formal"]),
        knowledge=PersonaLayer(),
        moral=PersonaLayer(traits=["just"]),
        belief=PersonaLayer(traits=["empirical"]),
    )

    text = render_persona(profile)

    assert text.startswith("Personality:\n- observant")
    assert "How you speak:\n- formal" in text
    assert "What you know" not in text


def test_repo_templates_only_use_known_placeholders():
    builder = make_builder()
    for path in ("prompts/generation/structured_prompt.txt", "prompts/generation/baseline_persona.txt"):
        builder.template = load_text(path)
        builder.render(builder.build("x"))