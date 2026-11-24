"""Non-simulation checks for exp176 settings, the van Rossum mapping and measurements."""

import numpy as np
import pytest
import torch
from experiments.exp176 import recipe
from experiments.exp176.analyse import (
    block_measures,
    dense_raster,
    rhythmicity,
    side_peak_lag,
)
from experiments.exp176.compute import presentation_steps, weight_summary
from snnlab.plasticity import NonlinearSTDP, SynapticScaling


def network_settings():
    cfg = recipe.configuration()
    cfg.update(
        dt_ms=0.1,
        n_input=6,
        n_e=4,
        n_i=2,
        n_output=3,
        tau_ampa_ms=2.0,
        tau_gaba_ms=6.0,
        readout_tau_ms=2.0,
        readout_threshold=1.0,
        surrogate_slope=1.0,
        neurons={
            name: dict(
                tau_mem_ms=20.0 if name == "E" else 5.0,
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


def test_configuration_rejects_unknown_and_invalid_settings():
    cfg = recipe.configuration()
    assert cfg["presentations"] == 50 and cfg["isi_block_presentations"] == 5
    assert cfg["accuracy_every"] == 5 and cfg["tracked_synapses"] == 1000
    with pytest.raises(ValueError):
        recipe.configuration(nonsense=1)
    with pytest.raises(ValueError):
        recipe.configuration(presentations=5)  # fewer than two ISI blocks
    with pytest.raises(ValueError):
        recipe.configuration(accuracy_images=10001)
    with pytest.raises(ValueError):
        recipe.configuration(device="cuda")


def test_van_rossum_mapping_places_the_fixed_point_at_the_trained_mean():
    cfg = recipe.configuration()
    weights = np.array([[0.0, 0.2], [0.6, 0.2]], dtype=np.float32)
    rule = recipe.rule_parameters(cfg, weights)
    c_d = cfg["stdp_depression_fraction"]
    assert rule["w_max_us"] == pytest.approx(0.6)
    assert rule["a_minus_us"] == pytest.approx(c_d * 0.6)  # depression = c_d * w
    assert rule["a_plus_us"] == pytest.approx(c_d * 0.25)  # potentiation = c_p
    # Uncorrelated drift a_plus*tau - a_minus*(w/w_max)*tau vanishes at the mean.
    w_star = rule["a_plus_us"] / rule["a_minus_us"] * rule["w_max_us"]
    assert w_star == pytest.approx(weights.mean())


def test_plasticity_request_is_the_van_rossum_pair():
    cfg = recipe.configuration()
    rule = recipe.rule_parameters(cfg, np.array([[0.1, 0.4]], dtype=np.float32))
    request = recipe.plasticity_request(cfg, rule, 12.5)
    stdp = request.rules[recipe.PLASTIC_PROJECTION]
    scaling = request.scaling[recipe.PLASTIC_PROJECTION]
    assert isinstance(stdp, NonlinearSTDP) and isinstance(scaling, SynapticScaling)
    assert (stdp.mu_plus, stdp.mu_minus) == (0.0, 1.0)
    assert stdp.tau_pre.value == stdp.tau_post.value == cfg["stdp_tau_ms"]
    assert scaling.target_rate.value == 12.5
    assert scaling.tau_rate.value == cfg["scaling_tau_rate_ms"]
    assert (stdp.w_min, stdp.w_max) == (scaling.w_min, scaling.w_max)
    with pytest.raises(ValueError):
        recipe.plasticity_request(cfg, rule, 0.0)


def test_networks_have_the_expected_projections():
    cfg = network_settings()
    for network, expected in (
        ("ping", {"input_to_E", "E_to_I", "I_to_E", "readout_projection"}),
        ("coba", {"input_to_E", "readout_projection"}),
    ):
        bundle = recipe.author_network(cfg, network)
        names = {p["id"] for p in bundle.graph["projections"]}
        assert names == expected
    source = {key: torch.ones(1) * i for i, key in enumerate(recipe.MAPPING.values())}
    assert set(recipe.native_weights(source, "coba")) == set(recipe.MAPPING_COBA)
    assert set(recipe.native_weights(source, "ping")) == set(recipe.MAPPING)


def test_accuracy_boundaries_cover_start_every_n_and_end():
    assert recipe.accuracy_boundaries(50, 5) == [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
    ]
    assert recipe.accuracy_boundaries(12, 5) == [0, 5, 10, 12]
    with pytest.raises(ValueError):
        recipe.accuracy_boundaries(50, 0)


def test_tracked_synapses_are_a_fixed_distinct_sample():
    picked = recipe.tracked_synapses((784, 1024), 1000, 176002)
    assert len(picked) == len(np.unique(picked)) == 1000
    assert picked.min() >= 0 and picked.max() < 784 * 1024
    np.testing.assert_array_equal(
        picked, recipe.tracked_synapses((784, 1024), 1000, 176002)
    )
    with pytest.raises(ValueError):
        recipe.tracked_synapses((2, 2), 5, 0)


def test_snapshot_boundaries_include_start_and_end():
    assert recipe.snapshot_indices(50, 5) == [0, 12, 25, 38, 50]
    assert recipe.snapshot_indices(4, 3) == [0, 2, 4]


def test_presentation_steps_accept_global_or_relative_and_reject_strays():
    base, steps = 4000, 2000
    np.testing.assert_array_equal(
        presentation_steps([4000, 5999], base, steps), [4000, 5999]
    )
    np.testing.assert_array_equal(
        presentation_steps([0, 1999], base, steps), [4000, 5999]
    )
    with pytest.raises(ValueError):
        presentation_steps([4000, 6000], base, steps)
    assert len(presentation_steps(np.empty(0), base, steps)) == 0


def test_weight_summary_tracks_change_and_memory():
    initial = np.array([[0.0, 1.0], [2.0, 3.0]])
    same = weight_summary(initial, initial, np.linspace(0, 3, 4))
    assert same["mean_abs_change"] == 0 and same[
        "correlation_with_initial"
    ] == pytest.approx(1)
    shifted = weight_summary(initial + 1, initial, np.linspace(0, 4, 5))
    assert shifted["net_change"] == pytest.approx(1)
    np.testing.assert_allclose(shifted["afferent_sum"], [4, 6])


def test_dense_raster_and_block_measures_recover_a_regular_train():
    cfg = recipe.configuration()
    cfg.update(dt_ms=0.1)
    times = np.arange(0, 1000, 25.0)  # one cell firing every 25 ms = 40 Hz
    cells = np.zeros(len(times), dtype=np.int16)
    raster = dense_raster(times, cells, 0, 1000, 0.1, 2)
    assert raster.shape == (10000, 2) and raster.sum() == len(times)
    measures, spectrum = block_measures(raster, cfg)
    assert measures["isi_median_ms"] == pytest.approx(25)
    assert measures["isi_cv_mean"] == pytest.approx(0, abs=1e-9)
    assert measures["rate_hz"] == pytest.approx(20)  # 40 Hz cell + silent cell
    assert measures["active_fraction"] == 0.5


def test_spectral_peak_finds_a_jittered_population_rhythm():
    cfg = recipe.configuration()
    cfg.update(dt_ms=0.1)
    rng = np.random.default_rng(0)
    cycles = np.arange(10, 1990, 25.0)  # 40 Hz volleys over 2 s
    cells = np.repeat(np.arange(200), len(cycles))
    times = np.tile(cycles, 200) + rng.normal(0, 1.5, cells.size)
    keep = (rng.random(cells.size) < 0.5) & (times >= 0) & (times < 2000)
    raster = dense_raster(times[keep], cells[keep], 0, 2000, 0.1, 200)
    measures, _ = block_measures(raster, cfg)
    assert measures["rhythm_peak_hz"] == pytest.approx(40, abs=4)
    assert measures["rhythm_fraction"] > 0.3


def test_accuracy_subset_matches_exp173():
    selected = recipe.test_sample_indices()
    np.testing.assert_array_equal(selected[:5], [6252, 4684, 1731, 4742, 4521])


def test_rhythmicity_contrast_separates_rhythm_from_noise():
    rng = np.random.default_rng(1)
    cycles = np.arange(5, 2000, 25.0)  # 40 Hz volleys
    cells = np.repeat(np.arange(200), len(cycles))
    times = np.tile(cycles, 200) + rng.normal(0, 1.0, cells.size)
    keep = (rng.random(cells.size) < 0.5) & (times >= 0) & (times < 2000)
    rhythmic = dense_raster(times[keep], cells[keep], 0, 2000, 0.1, 200)
    flat = (rng.random((20000, 200)) < 0.002).astype(np.uint8)  # 20 Hz Poisson
    sharp, _, _ = rhythmicity(rhythmic, 0.1)
    noise, _, _ = rhythmicity(flat, 0.1)
    assert sharp["rhythm_contrast"] > 0.5
    assert sharp["rhythm_period_ms"] == pytest.approx(25, abs=1.5)  # 40 Hz
    assert sharp["rhythm_frequency_hz"] == pytest.approx(40, abs=3)
    assert noise["rhythm_contrast"] is None or noise["rhythm_contrast"] < 0.1
    assert noise["rhythm_period_ms"] is None


def test_side_peak_ignores_rounding_noise_in_a_silent_trough():
    lags = np.arange(0, 41, 1.0)
    ac = np.zeros_like(lags)
    ac[1:4] = [5.0, 2.5, 0.8]
    ac[22:29] = [0.8, 2.6, 5.4, 6.9, 5.4, 2.6, 0.8]  # side peak at 25 ms
    rng = np.random.default_rng(0)
    ac[5:21] = rng.normal(0, 1e-17, 16)  # FFT rounding noise where firing is silent
    ac[0] = np.nan
    assert side_peak_lag(lags, ac, 8.0) == pytest.approx(25, abs=0.5)
    assert (
        side_peak_lag(lags, np.ones_like(lags) + rng.normal(0, 1e-3, lags.size), 3.0)
        is None
    )
