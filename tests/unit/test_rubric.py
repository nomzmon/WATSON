import pytest
import yaml
from pydantic import ValidationError

from watson.common.rubric import CriterionJudgement, load_rubric_weights, weighted_score


def test_weighted_score_sums_weights_times_scores():
    assert weighted_score({"A": 10, "B": 5}, {"A": 0.6, "B": 0.4}) == 8.0


def test_load_rubric_weights_returns_criteria_in_declared_order(tmp_path):
    path = tmp_path / "rubric.yaml"
    path.write_text(yaml.safe_dump({"weights": {"B": 0.4, "A": 0.6}}))

    assert list(load_rubric_weights(path, ("A", "B"), "TEST")) == ["A", "B"]


def test_load_rubric_weights_names_the_rubric_in_errors(tmp_path):
    path = tmp_path / "rubric.yaml"
    path.write_text(yaml.safe_dump({"weights": {"A": 0.6, "B": 0.6}}))

    with pytest.raises(ValueError, match="TEST weights must sum to 1.0"):
        load_rubric_weights(path, ("A", "B"), "TEST")


def test_criterion_judgement_enforces_1_to_10():
    with pytest.raises(ValidationError):
        CriterionJudgement(reasoning="x", score=0)