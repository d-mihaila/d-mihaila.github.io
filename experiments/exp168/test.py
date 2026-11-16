"""Focused checks; no simulation or upstream stage execution."""

import json

import pytest
from experiments.exp168 import recipe
from pingstore.library import Store


def test_configuration_contract():
    on_ai = recipe.configuration("AI", recurrence="on")
    on_ping = recipe.configuration("PING", recurrence="on")
    assert (on_ai["dt_ms"], on_ai["duration_ms"]) == (0.05, 2050.0)
    assert {k for k in on_ai if on_ai[k] != on_ping[k]} == {
        "condition",
        "external_rate_hz",
        "input_connection_probability",
        "input_i_gain",
    }
    assert on_ai["initializer_parent_means_us"]["E_to_E"] > 0
    assert on_ai["initializer_parent_means_us"]["I_to_I"] > 0
    for bad in (
        {"condition": "gamma"},
        {"condition": "AI", "recurrence": "masked"},
        {"condition": "AI", "device": "bad"},
        {"condition": "AI", "seed": -1},
    ):
        with pytest.raises(ValueError):
            recipe.configuration(**bad)


def test_native_on_evidence_contract():
    origin = Store().get(recipe.SOURCES["on"]["run_id"])
    configurations = recipe.case_configuration(origin, "on")
    assert set(configurations) == {"AI", "PING"}
    for state in recipe.CONDITIONS:
        snapshot = json.loads(
            recipe.case_file(origin, "on", state, "start", "snapshot.json").read_text()
        )
        projections = {p["id"] for p in snapshot["network"]["projections"]}
        assert {"E_to_E", "I_to_I"} <= projections


def test_report_retains_comparisons_and_gain_transition():
    """Validate selected scientific evidence without running a simulation."""
    store = Store()
    presented = store.get(store.latest(recipe.SLUG, "present")["id"])
    comparison_reference = presented.details["inputs"].get("comparison_figures")
    baseline = store.get(
        comparison_reference["run_id"]
        if comparison_reference
        else "exp168-r040-present"
    )
    numbers = json.loads(presented.file("numbers.json").read_text())
    original = json.loads(baseline.file("numbers.json").read_text())
    for key in ("configurations", "metrics"):
        assert numbers[key] == original[key]
    for recurrence in recipe.RECURRENT_CASES:
        for extension in ("svg", "png", "pdf"):
            name = f"recurrence-{recurrence}-rasters.{extension}"
            if comparison_reference:
                assert (
                    presented.file(name).read_bytes()
                    == baseline.file(name).read_bytes()
                )
    transition = numbers["gain_transition"]
    cfg = transition["configuration"]
    assert cfg["recurrence"] == "on"
    assert cfg["input_connection_probability"] == 0.05
    assert cfg["external_rate_hz"] == 2.5
    assert cfg["input_e_gain"] == 1.0
    assert cfg["input_i_gain"] == dict(
        before=1.0, after=0.0, start_ms=1050.0, end_ms=1550.0
    )
    assert transition["metrics"]["baseline"]["E_median_cv"] is None
    assert transition["metrics"]["hold"]["E_rhythmicity"] > 0.99
    assert transition["metrics"]["hold"]["mean_correlation"] > 0.3
    analysed = store.get(presented.details["inputs"]["analyse"]["run_id"])
    computed = store.get(analysed.details["inputs"]["transition_compute"]["run_id"])
    graph = json.loads(
        recipe.transition_file(computed, "network.bundle", "graph.json").read_text()
    )
    assert {"E_to_E", "I_to_I"} <= {p["id"] for p in graph["projections"]}
