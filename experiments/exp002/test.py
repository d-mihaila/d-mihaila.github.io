"""Non-simulation checks for exp002 settings, rules, streams and measurements."""

import numpy as np
import pytest
import torch
from experiments.exp002 import recipe
from experiments.exp002.analyse import (
    cycles,
    dense_raster,
    rhythmicity,
    side_peak_lag,
)
from experiments.exp002.compute import presentation_steps, weight_summary
from snnlab.plasticity import NonlinearSTDP, SynapticScaling


def network_settings(architecture="ping"):
    cfg = recipe.configuration()
    cfg.update(
        dt_ms=0.1,
        n_input=6,
        n_e=4,
        n_i=2,
        n_output=3,
        tau_ampa_ms=2.0,
        readout_tau_ms=2.0,
        readout_threshold=1.0,
        surrogate_slope=1.0,
        networks={"net": dict(architecture=architecture, tau_gaba_ms=6.0)},
        neurons={
            name: dict(
                tau_mem_ms=20.0,
                capacitance_nf=1.0,
                leak_us=0.05,
                resting_mv=-65.0,
                threshold_mv=-50.0,
                reset_mv=-65.0,
                initial_voltage_mv=-65.0,
                refractory_ms=1.2,
            )
            for name in ("E", "I")
        },
    )
    return cfg


def test_configuration_defaults_and_validation():
    cfg = recipe.configuration()
    assert cfg["source_cells"] == {
        "ping": "ping__rt25hz__seed42",
        "coba": "coba__rt25hz__seed42",
    }
    assert cfg["presentations"] == 50 and cfg["block_presentations"] == 5
    assert cfg["accuracy_every"] == 10 and cfg["accuracy_images"] == 500
    for bad in (
        dict(nonsense=1),
        dict(presentations=5),
        dict(accuracy_images=1001),
        dict(presentation_steps=10),
        dict(device="cuda"),
    ):
        with pytest.raises(ValueError):
            recipe.configuration(**bad)


def test_conditions_cross_networks_with_all_four_rules():
    conditions = recipe.conditions(recipe.configuration())
    assert [c["id"] for c in conditions] == [
        f"{n}__{r}" for n in ("ping", "coba") for r in recipe.RULES
    ]
    assert all(c["stream"] == "balanced" and c["setpoint"] == 1.0 for c in conditions)


def test_rule_mapping_keeps_the_fixed_point_and_window_area():
    cfg = recipe.configuration()
    weights = np.array([[0.0, 0.2], [0.6, 0.2]], dtype=np.float32)
    ref = recipe.rule_parameters(cfg, weights, 20.0)
    short = recipe.rule_parameters(cfg, weights, 5.0)
    assert ref["a_minus_us"] == pytest.approx(0.003 * 0.6)
    assert ref["a_plus_us"] / ref["a_minus_us"] * ref["w_max_us"] == pytest.approx(0.25)
    # Four times shorter window, four times larger amplitudes: same area.
    assert short["a_plus_us"] * 5 == pytest.approx(ref["a_plus_us"] * 20)
    assert short["equilibrium_us"] == ref["equilibrium_us"]


def test_each_rule_builds_the_expected_plasticity_request():
    cfg = recipe.configuration()
    params = recipe.rule_parameters(cfg, np.array([[0.1, 0.4]], np.float32), 20.0)
    assert recipe.plasticity_request(cfg, "frozen", params, 10.0) is None
    stdp = recipe.plasticity_request(cfg, "stdp", params, 10.0)
    assert isinstance(stdp.rules["input_to_E"], NonlinearSTDP) and not stdp.scaling
    assert (stdp.rules["input_to_E"].mu_plus, stdp.rules["input_to_E"].mu_minus) == (
        0.0,
        1.0,
    )
    scaling = recipe.plasticity_request(cfg, "scaling", params, 10.0)
    assert not scaling.rules and isinstance(
        scaling.scaling["input_to_E"], SynapticScaling
    )
    both = recipe.plasticity_request(cfg, "both", params, 12.5)
    assert set(both.rules) == set(both.scaling) == {"input_to_E"}
    assert both.scaling["input_to_E"].target_rate.value == 12.5
    with pytest.raises(ValueError):
        recipe.plasticity_request(cfg, "scaling", params, 0.0)


def test_stream_pool_avoids_the_accuracy_subset_and_is_reproducible():
    pool = recipe.stream_pool(10000, 177001)
    held_out = recipe.test_sample_indices(samples=1000)
    assert len(pool) == 9000 and not np.intersect1d(pool, held_out).size
    np.testing.assert_array_equal(pool, recipe.stream_pool(10000, 177001))
    # The 500-image accuracy subset is the first half of exp173's 1,000.
    np.testing.assert_array_equal(
        recipe.test_sample_indices(samples=500), held_out[:500]
    )
    labels = np.arange(10000) % 10
    plan = recipe.stream_plan(recipe.configuration(), "balanced", pool, labels, 25.0)
    assert (
        len(plan) == 50 and plan[0]["image"] == pool[0] and plan[0]["rate_hz"] == 25.0
    )


def test_networks_have_the_expected_projections():
    for architecture, expected in (
        ("ping", {"input_to_E", "E_to_I", "I_to_E", "readout_projection"}),
        ("coba", {"input_to_E", "readout_projection"}),
    ):
        bundle = recipe.author_network(network_settings(architecture), "net")
        assert {p["id"] for p in bundle.graph["projections"]} == expected
    source = {key: torch.ones(1) for key in recipe.MAPPING.values()}
    assert len(recipe.native_weights(source, "coba")) == 2
    assert len(recipe.native_weights(source, "ping")) == 4


def test_boundaries_and_tracked_synapses():
    assert recipe.accuracy_boundaries(50, 10) == [0, 10, 20, 30, 40, 50]
    assert recipe.snapshot_indices(50, 5) == [0, 12, 25, 38, 50]
    picked = recipe.tracked_synapses((784, 1024), 1000, 177003)
    assert len(np.unique(picked)) == 1000


def test_presentation_steps_and_weight_summary():
    np.testing.assert_array_equal(presentation_steps([0, 1], 4000, 2000), [4000, 4001])
    with pytest.raises(ValueError):
        presentation_steps([6000], 4000, 2000)
    initial = np.array([[0.0, 1.0], [2.0, 3.0]])
    same = weight_summary(initial, initial, np.linspace(0, 3, 4))
    assert same["mean_abs_change"] == 0 and same["skew"] == pytest.approx(0)


def ping_like(rng, period_ms=10.0, duration_ms=1000.0, e_cells=200, share=0.3):
    """I volleys every period; a random share of E cells fires just before each."""
    volleys = np.arange(5.0, duration_ms - 1, period_ms)
    i_times = np.repeat(volleys, 20) + rng.normal(0, 0.2, 20 * len(volleys))
    i_cells = np.tile(np.arange(20), len(volleys))
    e_times, e_cells_out = [], []
    for v in volleys:
        chosen = np.flatnonzero(rng.random(e_cells) < share)
        e_times.append(v - 1.5 + rng.normal(0, 0.3, len(chosen)))
        e_cells_out.append(chosen)
    e_times = np.concatenate(e_times)
    e_cells_out = np.concatenate(e_cells_out)
    cfg = dict(recipe.configuration(), dt_ms=0.1)
    e = dense_raster(e_times, e_cells_out, 0, duration_ms, 0.1, e_cells)
    i = dense_raster(i_times, i_cells, 0, duration_ms, 0.1, 20)
    return cfg, e, i


def test_cycles_measure_volley_rate_and_participation():
    cfg, e, i = ping_like(np.random.default_rng(3))
    measured = cycles(e, i, cfg)
    assert measured["cycle_rate_hz"] == pytest.approx(100, abs=3)
    assert measured["participation"] == pytest.approx(0.3, abs=0.03)
    assert cycles(e, None, cfg)["participation"] is None


def test_rhythmicity_separates_rhythm_from_noise():
    rng = np.random.default_rng(1)
    cfg, e, _ = ping_like(rng, period_ms=25.0, duration_ms=2000.0)
    sharp, _, _ = rhythmicity(e, cfg)
    flat = (rng.random((20000, 200)) < 0.002).astype(np.uint8)
    noise, _, _ = rhythmicity(flat, cfg)
    assert sharp["rhythm_contrast"] > 0.5
    assert sharp["rhythm_frequency_hz"] == pytest.approx(40, abs=3)
    assert noise["rhythm_contrast"] < 0.1 and noise["rhythm_period_ms"] is None


def test_side_peak_ignores_rounding_noise_in_a_silent_trough():
    lags = np.arange(0, 41, 1.0)
    ac = np.zeros_like(lags)
    ac[1:4] = [5.0, 2.5, 0.8]
    ac[22:29] = [0.8, 2.6, 5.4, 6.9, 5.4, 2.6, 0.8]
    ac[5:21] = np.random.default_rng(0).normal(0, 1e-17, 16)
    ac[0] = np.nan
    assert side_peak_lag(lags, ac, 8.0) == pytest.approx(25, abs=0.5)
