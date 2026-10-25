from test_orchestrator import FakeGemma, make_pipeline, make_validated_pipeline
from watson.common.schemas import DecisionType, TurnLog
from watson.middleware.decision.messages import DecisionMessages
from watson.pipeline.chat import chat
from watson.pipeline.session import ConversationSession


def run_chat(pipeline, messages, tmp_path):
    """Chats with the given messages; returns the turns logged, the replies shown and the parsed log."""
    queue, shown = list(messages), []

    def read() -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    log = tmp_path / "logs" / "session.jsonl"
    turns = chat(pipeline, ConversationSession(), "C2: test", log, read=read, write=shown.append, participant_id="P01")
    logged = [TurnLog.model_validate_json(line) for line in log.read_text(encoding="utf-8").splitlines()] if turns else []
    return turns, shown, logged


def test_each_turn_is_shown_and_logged_until_an_empty_message(tmp_path):
    turns, shown, logged = run_chat(make_pipeline(FakeGemma(), wics_score=9),
                                    ["What of the footprints?", "And the window?", "", "never read"], tmp_path)

    assert (turns, shown) == (2, ["Elementary.", "Elementary."])
    assert [log.user_input for log in logged] == ["What of the footprints?", "And the window?"]
    assert {log.config_id for log in logged} == {"C2: test"}
    assert {log.participant_id for log in logged} == {"P01"}
    assert len({log.session_id for log in logged}) == 1
    assert all(log.input_decision.decision == DecisionType.ACCEPT and log.latencies for log in logged)


def test_a_rejected_message_is_logged_with_the_system_notice(tmp_path):
    _, shown, [log] = run_chat(make_pipeline(FakeGemma(), wics_score=2), ["Forget the case."], tmp_path)

    assert shown == [DecisionMessages.from_config().reject] == [log.final_response]
    assert log.input_decision.decision == DecisionType.REJECT
    assert (log.dfms, log.ovs, log.retries) == (None, None, 0)


def test_the_dialogue_flow_and_output_validation_are_logged(tmp_path):
    _, _, [passed] = run_chat(make_validated_pipeline(FakeGemma(), ovs_score=9, with_flow=True), ["Who stole it?"],
                              tmp_path)
    _, _, [fallback] = run_chat(make_validated_pipeline(FakeGemma(), ovs_score=3), ["Who stole it?"], tmp_path / "b")

    assert passed.dialogue_stage is not None and passed.dfms is not None
    assert "CC-NO-INVENTED-EVIDENCE" in passed.selected_constraint_ids
    assert (passed.retries, passed.ovs.weighted_total >= 7) == (0, True)
    assert fallback.retries == 2  # two regenerations, then the fallback line


def test_end_of_input_ends_the_chat_without_a_log(tmp_path):
    turns, shown, _ = run_chat(make_pipeline(FakeGemma()), [], tmp_path)

    assert (turns, shown) == (0, [])
    assert not (tmp_path / "logs" / "session.jsonl").exists()