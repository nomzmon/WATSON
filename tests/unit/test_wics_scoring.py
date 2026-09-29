import pytest
import yaml

from watson.middleware.input_constraint.scoring import compute_wics, load_wics_weights

THEORY_WEIGHTS = {"OOP": 0.35, "TR": 0.25, "HC": 0.25, "GIP": 0.15}


def test_repo_config_has_theory_driven_weights():
    assert load_wics_weights() == THEORY_WEIGHTS


def test_compute_wics_applies_equation_5_1():
    scores = {"OOP": 10, "TR": 8, "HC": 6, "GIP": 4}
    assert compute_wics(scores, THEORY_WEIGHTS) == 7.6


def test_boundary_score_is_not_lost_to_float_error():
    scores = {"OOP": 8, "TR": 8, "HC": 8, "GIP": 8}
    assert compute_wics(scores, THEORY_WEIGHTS) == 8.0


def test_weights_must_cover_all_four_criteria(tmp_path):
    path = tmp_path / "wics.yaml"
    path.write_text(yaml.safe_dump({"weights": {"OOP": 0.5, "TR": 0.5}}))

    with pytest.raises(ValueError, match="exactly"):
        load_wics_weights(path)


def test_weights_must_sum_to_one(tmp_path):
    path = tmp_path / "wics.yaml"
    path.write_text(yaml.safe_dump({"weights": {"OOP": 0.5, "TR": 0.5, "HC": 0.5, "GIP": 0.5}}))

    with pytest.raises(ValueError, match="sum to 1.0"):
        load_wics_weights(path)