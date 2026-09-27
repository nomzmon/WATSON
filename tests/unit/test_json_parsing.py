import pytest

from watson.llm.json_parsing import JSONParseError, parse_json_object


def test_parses_plain_json():
    assert parse_json_object('{"score": 4}') == {"score": 4}


def test_parses_json_inside_code_fence():
    text = 'Here are the scores:\n```json\n{"score": 4}\n```'
    assert parse_json_object(text) == {"score": 4}


def test_parses_json_surrounded_by_prose():
    text = 'Sure! {"score": 4, "rationale": "in character"} Hope that helps.'
    assert parse_json_object(text) == {"score": 4, "rationale": "in character"}


def test_repairs_trailing_commas():
    assert parse_json_object('{"scores": [1, 2,], "total": 3,}') == {"scores": [1, 2], "total": 3}


def test_rejects_output_without_a_json_object():
    with pytest.raises(JSONParseError):
        parse_json_object("I cannot score this.")


def test_rejects_json_that_is_not_an_object():
    with pytest.raises(JSONParseError):
        parse_json_object("[1, 2, 3]")