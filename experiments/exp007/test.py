"""Non-simulation checks for exp007 settings, rules, streams and measurements."""

import numpy as np
import pytest
import torch
from experiments.exp007 import recipe
from experiments.exp007.analyse import (
    cycles,
    dense_raster,
    ranks,
    rhythmicity,
    side_peak_lag,
    spearman,
    tendency,
    weight_maps,
)
from experiments.exp007.compute import presentation_steps, weight_summary
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
    digits = cfg["digits"]
    assert len(digits) == 3 and len(set(digits)) == 3
    assert all(0 <= d <= 9 for d in digits)
    assert recipe.configuration()["digits"] == digits  # seeded, reproducible
    assert recipe.configuration(digits=[4, 2])["digits"] == [4, 2]
    for bad in (
        dict(nonsense=1),
        dict(presentations=5),
        dict(digits=[3, 3]),
        dict(digits=[10]),
        dict(digits=[]),
        dict(digit_count=11),
        dict(digit_pixel_threshold=1.5),
        dict(change_tolerance_fraction=0),
        dict(presentation_steps=10),
        dict(device="cuda"),
    ):
        with pytest.raises(ValueError):
            recipe.configuration(**bad)


def test_conditions_cross_networks_and_digits_with_frozen_first():
    cfg = recipe.configuration(digits=[7, 1, 4])
    conditions = recipe.conditions(cfg)
    assert len(conditions) == 18
    assert [c["id"] for c in conditions[:4]] == [
        "ping__digit7__frozen",
        "ping__digit7__stdp",
        "ping__digit7__both",
        "ping__digit1__frozen",
    ]
    assert conditions[-1]["id"] == "coba__digit4__both"
    # The digits, images and spike trains are exp006's.
    assert recipe.configuration()["digits"] == [2, 6, 0]
    assert {c["stream"] for c in conditions} == {"digit7", "digit1", "digit4"}
    assert all(c["digit"] == recipe.stream_digit(c["stream"]) for c in conditions)
    with pytest.raises(ValueError):
        recipe.stream_digit("long")


def test_stream_plan_repeats_one_image_of_the_digit():
    cfg = recipe.configuration(digits=[5])
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 9000)
    pool = rng.permutation(9000)
    plan = recipe.stream_plan(cfg, "digit5", pool, labels, 30.0)
    first = next(int(i) for i in pool if labels[i] == 5)
    assert len(plan) == 50
    assert {item["image"] for item in plan} == {first}
    assert {item["label"] for item in plan} == {5}
    assert {item["rate_hz"] for item in plan} == {30.0}
    assert recipe.setpoint_window(cfg, "digit5") == (0, 50)
    with pytest.raises(ValueError):
        recipe.stream_plan(cfg, "digit5", pool, np.zeros(9000, int), 30.0)
    mask = recipe.digit_mask(np.array([0.0, 0.2, 0.3, 1.0]), 0.25)
    np.testing.assert_array_equal(mask, [False, False, True, True])


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
    pool = recipe.stream_pool(10000, 181002)
    held_out = recipe.test_sample_indices(samples=1000)
    assert len(pool) == 9000 and not np.intersect1d(pool, held_out).size
    np.testing.assert_array_equal(pool, recipe.stream_pool(10000, 181002))


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
    assert recipe.snapshot_indices(50, 6) == [0, 10, 20, 30, 40, 50]
    picked = recipe.tracked_synapses((784, 1024), 1000, 181004)
    assert len(np.unique(picked)) == 1000


def test_presentation_steps_and_weight_summary():
    np.testing.assert_array_equal(presentation_steps([0, 1], 4000, 2000), [4000, 4001])
    with pytest.raises(ValueError):
        presentation_steps([6000], 4000, 2000)
    initial = np.array([[0.0, 1.0], [2.0, 3.0], [1.0, 1.0]])
    mask = np.array([True, False, False])  # row 0 = the digit's pixel
    same = weight_summary(initial, initial, np.linspace(0, 3, 4), mask, 0.01)
    assert same["mean_abs_change"] == 0 and same["digit_potentiated"] == 0
    np.testing.assert_allclose(same["pixel_mean"], [0.5, 2.5, 1.0])
    later = initial + np.array([[0.5, 0.5], [-0.5, 0.0], [0.0, 0.005]])
    moved = weight_summary(later, initial, np.linspace(0, 3, 4), mask, 0.01)
    assert moved["digit_net_change"] == pytest.approx(0.5)
    assert moved["digit_potentiated"] == 1 and moved["digit_depressed"] == 0
    assert moved["other_net_change"] == pytest.approx(-0.12375)
    # One of four other synapses fell; the 0.005 rise is inside the tolerance.
    assert moved["other_depressed"] == 0.25 and moved["other_potentiated"] == 0


def test_tendency_and_maps_split_pixel_groups():
    initial = np.array([[0.1, 0.9], [0.1, 0.9]])
    final = initial + np.array([[0.1, -0.1], [0.0, 0.0]])
    mask = np.array([True, False])
    edges, binned = tendency(initial, final, mask, 1.0, 2)
    np.testing.assert_allclose(edges, [0, 0.5, 1])
    np.testing.assert_allclose(binned["digit"][0], [0.1, -0.1])  # weak up, strong down
    np.testing.assert_allclose(binned["other"][0], [0.0, 0.0])
    maps = weight_maps(initial, final, mask, 1.0, 4)
    assert maps["joint_digit"].sum() == 2 and maps["ratio_other"].shape == (2, 4)
    assert maps["edges_digit"][-1] <= 1.0


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


def test_rank_correlation():
    np.testing.assert_allclose(ranks([3.0, 1.0, 1.0, 2.0]), [3.0, 0.5, 0.5, 2.0])
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def synthetic_context(tmp_path):
    """Random results in the shapes analyse writes, for every figure (no simulation)."""
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    n_e, n_i, windows, block, bins, lags = 8, 4, 10, 5, 6, 40
    digits = [2, 6]
    cfg = dict(
        recipe.configuration(
            digits=digits, presentations=windows, block_presentations=block
        ),
        n_e=n_e,
        n_i=n_i,
        presentation_ms=200.0,
        networks={
            "ping": dict(architecture="ping", cell="ping__rt25hz__seed42"),
            "coba": dict(architecture="coba", cell="coba__rt25hz__seed42"),
        },
    )
    cfg["conditions"] = recipe.conditions(cfg)
    edges_isi = np.logspace(0, np.log10(2000), 41)
    blocks = windows // block
    block_keys = (
        "rhythm_contrast",
        "rhythm_frequency_hz",
        "cycle_rate_hz",
        "participation",
        "rate_hz",
        "isi_cv_mean",
        "isi_median_ms",
        "spectral_peak_hz",
    )

    def block_rows():
        return [
            dict(
                {k: float(rng.random()) for k in block_keys},
                isi_histogram=rng.integers(0, 9, 40).tolist(),
            )
            for _ in range(blocks)
        ]

    results, series, spikes, rules, summary = {}, {}, {}, {}, {}
    for c in cfg["conditions"]:
        key, plastic = c["id"], c["rule"] != "frozen"
        results[key] = dict(
            presentations=windows,
            rates_hz={"E": rng.random(windows).tolist()},
            blocks={"E": block_rows(), "I": block_rows()},
        )
        rules[key] = dict(w_max_us=1.0, equilibrium_us=0.3, target_rate_hz=20.0)
        summary[key] = {
            f"{k}_{e}": float(rng.random())
            for k in block_keys
            for e in ("first", "last")
        }
        times = np.sort(rng.random(400) * windows * 200.0).astype(np.float32)
        spikes[key] = dict(
            E_time_ms=times,
            E_cell=rng.integers(0, n_e, 400).astype(np.int16),
            I_time_ms=times,
            I_cell=rng.integers(0, n_i, 400).astype(np.int16),
        )
        series.update(
            {
                f"{key}__E_autocorrelogram_lags_ms": np.arange(lags, dtype=float),
                f"{key}__E_autocorrelogram": 1 + rng.random((blocks, lags)),
                f"{key}__E_spectrum_frequencies_hz": np.linspace(0, 500, 50),
                f"{key}__E_spectrum_power": 1e-3 + rng.random((blocks, 50)),
                f"{key}__E_cell_rate_first": 40 * rng.random(n_e),
                f"{key}__E_cell_rate_last": 40 * rng.random(n_e),
            }
        )
        if not plastic:
            continue
        curve = lambda: rng.random(windows + 1).tolist()  # noqa: E731
        results[key]["weights"] = dict(
            mean_us=curve(),
            mean_abs_change_us=curve(),
            correlation_with_initial=curve(),
            skew=curve(),
            scaling_rate_mean_hz=rng.random(windows).tolist()
            if c["rule"] == "both"
            else None,
            **{
                f"{g}_{k}": curve()
                for g in ("digit", "other")
                for k in (
                    "mean",
                    "net_change",
                    "abs_change",
                    "potentiated",
                    "depressed",
                )
            },
        )
        summary[key].update(
            digit_net_change_last=0.1,
            other_net_change_last=-0.05,
            weight_change_last=0.2,
            weight_correlation_last=0.9,
        )
        mask = np.zeros(784, bool)
        mask[:100] = True
        series.update(
            {
                f"{key}__tendency_{g}": rng.normal(size=(3, 20))
                for g in ("digit", "other")
            }
            | {
                f"{key}__tendency_counts_{g}": np.full(20, 50)
                for g in ("digit", "other")
            }
            | {
                f"{key}__tendency_edges": np.linspace(0, 1, 21),
                f"{key}__snapshot_boundaries": np.array([4, 7, 10]),
                f"{key}__tracked_weights": rng.random((windows + 1, 30)),
                f"{key}__tracked_index": rng.choice(784 * n_e, 30, replace=False),
                f"{key}__digit_mask": mask,
                f"{key}__afferent_first": 1 + rng.random(n_e),
                f"{key}__afferent_last": 1 + rng.random(n_e),
            }
            | {
                f"{key}__joint_{g}": rng.integers(0, 5, (bins, bins))
                for g in ("digit", "other")
            }
            | {
                f"{key}__ratio_{g}": rng.integers(1, 5, (2, bins))
                for g in ("digit", "other")
            }
            | {
                f"{key}__edges_{g}": np.linspace(0, 0.8, bins + 1)
                for g in ("digit", "other")
            }
        )
    effects = {}
    for network in cfg["source_cells"]:
        for digit in digits:
            key = f"{network}__digit{digit}"
            effects[key] = dict(
                digit_extra_change=0.01, other_extra_change=-0.02, setpoint_hz=20.0
            )
            for g in ("digit", "other"):
                series[f"{key}__scaling_joint_{g}"] = rng.integers(0, 5, (bins, bins))
                series[f"{key}__scaling_edges_{g}"] = np.linspace(-0.1, 0.1, bins + 1)
            series[f"{key}__afferent_ratio"] = 0.5 + rng.random(n_e)
            series[f"{key}__stdp_rate_over_setpoint"] = 2 * rng.random(n_e)
    numbers = dict(
        rules=rules,
        summary=summary,
        scaling_effect=effects,
        isi_edges_ms=edges_isi.tolist(),
    )
    stimulus = dict(
        stimulus_images=np.array([11, 12]),
        stimulus_digits=np.array(digits),
        stimulus_pixels=rng.random((2, 784)),
    )
    return SimpleNamespace(
        export=tmp_path,
        numbers=numbers,
        cfg=cfg,
        results=results,
        series=series,
        spikes=spikes,
        stimulus=stimulus,
    )


def test_every_figure_renders_from_synthetic_results(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from experiments.exp007 import present

    ctx = synthetic_context(tmp_path)
    for figure in present.FIGURES:
        figure(ctx)
    names = {f.__name__.removeprefix("figure_") for f in present.FIGURES}
    written = {path.stem for path in tmp_path.glob("*.svg")}
    assert names <= written, names - written
