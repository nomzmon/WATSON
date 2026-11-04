from datetime import datetime

from test_middleware_regression import run
from watson.common.config import load_yaml
from watson.common.rubric import load_rubric_weights
from watson.evaluation.weight_comparison import compare_weights, render_weight_comparison
from watson.middleware.dialogue_flow.flow_evaluator import DFMS_CRITERIA, load_dfms_weights
from watson.middleware.input_constraint.scoring import WICS_CRITERIA, load_wics_weights
from watson.middleware.output_validation.validator import OVS_CRITERIA, load_ovs_weights

EQUAL = {name: load_rubric_weights(f"configs/rubrics/variants/{name.lower()}_equal.yaml", criteria, name)
         for name, criteria in (("WICS", WICS_CRITERIA), ("DFMS", DFMS_CRITERIA), ("OVS", OVS_CRITERIA))}


def complete_framework_run():
    return run(5)  # the middleware regression's last step has every component, output validation included


def test_the_equal_variants_give_every_criterion_the_same_weight():
    assert all(set(weights.values()) == {0.25} for weights in EQUAL.values())


def test_the_equal_variants_match_the_current_rubrics_except_for_the_weights():
    for name in ("wics", "dfms", "ovs"):
        current, variant = load_yaml(f"configs/rubrics/{name}.yaml"), load_yaml(f"configs/rubrics/variants/{name}_equal.yaml")
        assert {k: v for k, v in variant.items() if k != "weights"} == {k: v for k, v in current.items() if k != "weights"}


def test_the_current_weights_as_the_variant_change_nothing():
    comparisons = compare_weights({"run": complete_framework_run()}, load_wics_weights(), load_dfms_weights(),
                                  load_ovs_weights())

    assert [(c.rubric, c.changed) for c in comparisons] == [("WICS", []), ("DFMS", []), ("OVS", [])]
    assert all(c.turns > 0 for c in comparisons)


def test_decisions_that_depend_on_the_weights_are_reported():
    records = complete_framework_run()
    wics_turn = next(r for r in records if r.wics_total is not None)
    ovs_turn = next(r for r in records if r.ovs_first_scores)
    # Every criterion is at least 7, so only the total decides: 8.05 with the current weights (accept), 7.75 equal.
    wics_turn = wics_turn.model_copy(update={"wics_oop": 10, "wics_tr": 7, "wics_hc": 7, "wics_gip": 7})
    # No criterion is below 5, so only the total decides: 7.5 with the current weights (pass), 6.75 equal (fail).
    ovs_turn = ovs_turn.model_copy(update={"ovs_first_scores": "PCC 10, NC 6, HWB 6, SS 5"})

    wics, _, ovs = compare_weights({"seed 7": [wics_turn, ovs_turn]}, EQUAL["WICS"], EQUAL["DFMS"], EQUAL["OVS"])

    assert wics.changed == [f"seed 7, step 5, {wics_turn.dialogue_id} turn {wics_turn.turn} ({wics_turn.kind.value}): "
                            "accept -> rephrase"]
    assert ovs.changed == [f"seed 7, step 5, {ovs_turn.dialogue_id} turn {ovs_turn.turn} ({ovs_turn.kind.value}): "
                           "pass -> fail"]


def test_the_report_shows_the_weights_counts_and_changes():
    comparisons = compare_weights({"run": complete_framework_run()}, EQUAL["WICS"], EQUAL["DFMS"], EQUAL["OVS"])

    report = render_weight_comparison(datetime(2026, 11, 4, 15, 0), ["experiments/x/runs/run"], "equal", comparisons)

    assert report.startswith("# Weight comparison: current weights against equal weights, 2026-11-04 15:00")
    assert "| WICS | input decision | OOP 0.35, TR 0.25, HC 0.25, GIP 0.15 | OOP 0.25, TR 0.25, HC 0.25, GIP 0.25 |" in report
    assert "## OVS: changed decisions" in report