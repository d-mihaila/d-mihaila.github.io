"""Measure rhythm, sparsity, rates, ISIs, weights and decoding for every condition."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
from experiments.exp002 import recipe
from pingstore.library import source_run, stage_run, write_json_atomic
from snnlab import analysis

ISI_EDGES_MS = np.logspace(0, np.log10(2000), 41)


def dense_raster(times_ms, cells, start_ms, stop_ms, dt_ms, n_cells):
    """Binary (time, cell) raster for one block of presentations."""
    steps = int(round((stop_ms - start_ms) / dt_ms))
    raster = np.zeros((steps, n_cells), dtype=np.uint8)
    keep = (times_ms >= start_ms) & (times_ms < stop_ms)
    index = np.rint((times_ms[keep] - start_ms) / dt_ms).astype(np.int64)
    raster[np.clip(index, 0, steps - 1), cells[keep].astype(np.int64)] = 1
    return raster


def side_peak_lag(lags, ac, trough_lag):
    """Lag of the autocorrelogram's first local maximum after the trough (the period).

    Same 3-point smoothing as the contrast. None unless the peak rises above the
    independence baseline of 1 and stands clearly above the trough,
    (peak − trough)/(peak + trough) >= 0.1. Parabolic interpolation refines it.
    """
    if trough_lag is None:
        return None
    values = np.asarray(ac, dtype=float).copy()
    values[0] = values[1]
    if not np.isfinite(values).all():
        return None
    smooth = np.convolve(values, [0.25, 0.5, 0.25], mode="same")
    trough = smooth[int(np.argmin(np.abs(np.asarray(lags) - trough_lag)))]
    for i in np.flatnonzero(np.asarray(lags) > trough_lag):
        if i + 1 >= len(smooth):
            break
        # First local maximum that rises above the independence baseline; tiny
        # rounding bumps in a silent trough (FFT noise) are skipped.
        if smooth[i] > 1 and smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]:
            peak = smooth[i]
            if (peak - trough) / (peak + trough) < 0.1:
                return None
            left, right = smooth[i - 1], smooth[i + 1]
            curvature = left - 2 * peak + right
            shift = 0.5 * (left - right) / curvature if curvature < 0 else 0.0
            return float(lags[i] + shift * (lags[1] - lags[0]))
    return None


def rhythmicity(raster, cfg):
    """Eoin's exp054 contrast (lobe − trough)/(lobe + trough), lags, period, curve."""
    dt, lag, width = cfg["dt_ms"], cfg["rhythm_lag_ms"], cfg["rhythm_bin_ms"]
    lags, ac = analysis.spike_autocorrelogram(raster, dt, lag, width)
    iei_lags, iei = analysis.iei_histogram(
        analysis.population_event_times(raster, dt), lag, width
    )
    scalars = analysis.rhythmicity_scalars(lags, ac, iei_lags, iei, width)
    period = side_peak_lag(lags, ac, scalars["trough_lag"])
    return (
        dict(
            rhythm_contrast=scalars["contrast"],
            lobe_lag_ms=scalars["lobe_lag"],
            trough_lag_ms=scalars["trough_lag"],
            rhythm_period_ms=period,
            rhythm_frequency_hz=None if period is None else 1000.0 / period,
        ),
        np.asarray(lags, dtype=float),
        np.asarray(ac, dtype=float),
    )


def cycles(e_raster, i_raster, cfg):
    """PING cycles from I volleys: cycle rate and the E cells taking part per cycle."""
    if i_raster is None or i_raster.sum() < 2:
        return dict(cycle_rate_hz=None, participation=None, spikes_per_active_cell=None)
    dt = cfg["dt_ms"]
    bursts = analysis.detect_population_bursts(
        i_raster, dt, smoothing_ms=1.0, min_separation_ms=cfg["cycle_min_separation_ms"]
    )
    seconds = e_raster.shape[0] * dt / 1000
    if bursts["status"] != "ok" or len(bursts["peak_steps"]) < 2:
        return dict(cycle_rate_hz=0.0, participation=None, spikes_per_active_cell=None)
    edges = analysis.cycle_boundaries(
        bursts["peak_steps"], e_raster.shape[0], policy="peak_to_peak"
    )["edges"]
    counts = analysis.cycle_spike_counts(e_raster, edges)["counts"]
    active = counts > 0
    spikes = counts[active]
    return dict(
        cycle_rate_hz=float(len(bursts["peak_steps"]) / seconds),
        participation=float(active.mean()),
        spikes_per_active_cell=float(spikes.mean()) if spikes.size else None,
    )


def spectrum_measures(raster, cfg):
    trace = analysis.population_rate_trace(
        raster, cfg["dt_ms"], bin_ms=cfg["rate_bin_ms"]
    )
    spectrum = analysis.power_spectrum(
        trace["rate_hz"],
        cfg["rate_bin_ms"],
        method="welch",
        nperseg=min(256, len(trace["rate_hz"])),
        detrend="constant",
    )
    band = cfg["rhythm_band_hz"]
    peak, share = None, None
    if spectrum["status"] == "ok":
        f, power = spectrum["frequencies_hz"], spectrum["power"]
        peak = analysis.spectral_peak(f, power, band)["frequency_hz"]
        inside = analysis.band_power(f, power, band)["power"]
        total = analysis.band_power(f, power, [1.0, 1000 / (2 * cfg["rate_bin_ms"])])[
            "power"
        ]
        share = inside / total if inside is not None and total else None
    return dict(spectral_peak_hz=peak, band_power_share=share), spectrum


def block_measures(raster, cfg, burn):
    """ISI, rhythmicity and spectral measures of one population in one block."""
    dt = cfg["dt_ms"]
    isi = analysis.inter_spike_intervals(raster, dt)
    pooled = np.concatenate(isi["intervals_ms"]) if isi["intervals_ms"] else np.empty(0)
    cv = analysis.isi_cv(raster, dt, aggregation="mean")
    rhythm, lags, ac = rhythmicity(raster[burn:], cfg)
    spectral, spectrum = spectrum_measures(raster, cfg)
    measures = dict(
        rate_hz=float(raster.sum() / raster.shape[1] / (raster.shape[0] * dt / 1000)),
        active_fraction=float((raster.sum(axis=0) > 0).mean()),
        isi_median_ms=float(np.median(pooled)) if len(pooled) else None,
        isi_cv_mean=cv["aggregate"],
        isi_histogram=np.histogram(pooled, bins=ISI_EDGES_MS)[0].tolist(),
        **rhythm,
        **spectral,
    )
    return measures, lags, ac, spectrum


def analyse_condition(source, cfg, condition):
    cid = condition["id"]
    architecture = cfg["networks"][condition["network"]]["architecture"]
    pops = recipe.populations(architecture)
    sizes = {"E": cfg["n_e"], "I": cfg["n_i"]}
    with np.load(source.file(cid, "spikes.npz"), allow_pickle=False) as spikes:
        data = {key: spikes[key] for key in spikes.files}
    total = len(data["E_counts"])
    seconds = cfg["presentation_ms"] / 1000
    block = cfg["block_presentations"]
    blocks = total // block
    result = dict(rates_hz={}, blocks={}, presentations=total)
    arrays = {}
    for p in pops:
        if data[p + "_counts"].sum() != len(data[p + "_cell"]):
            raise ValueError("Spike evidence does not match its counts")
        result["rates_hz"][p] = (data[p + "_counts"] / sizes[p] / seconds).tolist()
    rows = {p: [] for p in pops}
    curves = {p: [] for p in pops}
    powers = {p: [] for p in pops}
    for b in range(blocks):
        start = b * block * cfg["presentation_ms"]
        stop = (b + 1) * block * cfg["presentation_ms"]
        burn = int(round(cfg["rhythm_burn_ms"] / cfg["dt_ms"])) if b == 0 else 0
        # Skip the start-up transient only when the first block is long enough.
        burn = burn if 2 * burn < (stop - start) / cfg["dt_ms"] else 0
        rasters = {
            p: dense_raster(
                data[p + "_time_ms"],
                data[p + "_cell"],
                start,
                stop,
                cfg["dt_ms"],
                sizes[p],
            )
            for p in pops
        }
        for p in pops:
            measures, lags, ac, spectrum = block_measures(rasters[p], cfg, burn)
            measures.update(start_ms=start, stop_ms=stop)
            if p == "E":
                measures.update(cycles(rasters["E"], rasters.get("I"), cfg))
                rate_in = data["stream_rates_hz"][b * block : (b + 1) * block].mean()
                measures["pairing_proxy_hz2"] = float(
                    rate_in * cfg["mean_pixel"] * measures["rate_hz"]
                )
            rows[p].append(measures)
            curves[p].append(ac)
            powers[p].append(np.asarray(spectrum["power"], dtype=float))
            arrays[f"{p}_autocorrelogram_lags_ms"] = lags
            arrays[f"{p}_spectrum_frequencies_hz"] = np.asarray(
                spectrum["frequencies_hz"], dtype=float
            )
    for p in pops:
        result["blocks"][p] = rows[p]
        arrays[f"{p}_autocorrelogram"] = np.stack(curves[p])
        arrays[f"{p}_spectrum_power"] = np.stack(powers[p])
    predicted = data["stream_scores"].argmax(axis=1)
    hits = (predicted == data["stream_labels"]).astype(float)
    result["online_decoding"] = [
        float(hits[b * block : (b + 1) * block].mean()) for b in range(blocks)
    ]
    with np.load(source.file(cid, "accuracy.npz"), allow_pickle=False) as values:
        boundaries, correct = values["boundaries"], values["correct"]
        predictions = values["predictions"]
    if correct.ndim != 2 or boundaries[0] != 0:
        raise ValueError("Accuracy evidence must start at the trained weights")
    images = correct.shape[1]
    curve = [float(100 * row.mean()) if images else None for row in correct]
    labels = np.load(source.file("dataset.npz"))["accuracy_labels"]
    per_class = [
        [
            float(100 * (row[labels == k] == k).mean()) if (labels == k).any() else None
            for k in range(10)
        ]
        for row in predictions
    ]
    result["accuracy"] = dict(
        boundaries=boundaries.tolist(),
        percent_curve=curve,
        per_class_curve=per_class,
        images=int(images),
        percent=curve[-1],
    )
    if condition["rule"] != "frozen":
        with np.load(source.file(cid, "weights.npz"), allow_pickle=False) as w:
            weights = {key: w[key] for key in w.files}
        result["weights"] = dict(
            mean_us=weights["mean"].tolist(),
            std_us=weights["std"].tolist(),
            skew=weights["skew"].tolist(),
            mean_abs_change_us=weights["mean_abs_change"].tolist(),
            net_change_us=weights["net_change"].tolist(),
            correlation_with_initial=weights["correlation_with_initial"].tolist(),
            scaling_rate_mean_hz=weights["scaling_rate_hz"].mean(axis=1).tolist()
            if weights["scaling_rate_hz"].size
            else None,
        )
        arrays.update(
            weight_histogram=weights["histogram"],
            weight_histogram_edges=weights["histogram_edges"],
            tracked_weights=weights["tracked_weights"],
        )
    return result, arrays


def summary(results, cfg):
    """First-versus-last-block numbers quoted in the report."""
    rows = {}
    keys = (
        "rhythm_contrast",
        "rhythm_frequency_hz",
        "cycle_rate_hz",
        "participation",
        "rate_hz",
        "isi_cv_mean",
        "spectral_peak_hz",
    )
    for cid, r in results.items():
        e = r["blocks"]["E"]
        row = {f"{k}_first": e[0].get(k) for k in keys}
        row.update({f"{k}_last": e[-1].get(k) for k in keys})
        if "I" in r["blocks"]:
            row.update(
                i_rate_first_hz=r["blocks"]["I"][0]["rate_hz"],
                i_rate_last_hz=r["blocks"]["I"][-1]["rate_hz"],
                i_contrast_first=r["blocks"]["I"][0]["rhythm_contrast"],
                i_contrast_last=r["blocks"]["I"][-1]["rhythm_contrast"],
            )
        row.update(
            accuracy_first=r["accuracy"]["percent_curve"][0],
            accuracy_last=r["accuracy"]["percent"],
            online_first=r["online_decoding"][0],
            online_last=r["online_decoding"][-1],
        )
        if "weights" in r:
            w, mean0 = r["weights"], r["weights"]["mean_us"][0]
            row.update(
                weight_change_last=w["mean_abs_change_us"][-1] / mean0,
                weight_mean_ratio_last=w["mean_us"][-1] / mean0,
                weight_correlation_last=w["correlation_with_initial"][-1],
                weight_skew_first=w["skew"][0],
                weight_skew_last=w["skew"][-1],
            )
        rows[cid] = row
    return rows


def analyse(identity):
    started = time.perf_counter()
    source = source_run(
        ROOT / ".pingstore",
        identity,
        experiment=recipe.source_experiment(identity),
        stage="compute",
    )
    cfg = source.details["execution"]["configuration"]
    rules = json.loads(source.file("rules.json").read_text())
    with np.load(source.file("dataset.npz"), allow_pickle=False) as dataset:
        cfg = dict(cfg, mean_pixel=None)
        pool = dataset["stream_pool"]
    # Mean pixel intensity of the first stream images: converts input rate to a
    # mean presynaptic rate for the pairing proxy (rate_in x mean pixel x E rate).
    from snnlab import sim

    _, pixels, _, _ = sim.load_dataset(
        "mnist", split=True, evaluation_only=True, evaluation_split="test"
    )
    cfg["mean_pixel"] = float(pixels[pool[: cfg["presentations"]]].mean())
    results, arrays = {}, {}
    for condition in cfg["conditions"]:
        tick = time.perf_counter()
        results[condition["id"]], extra = analyse_condition(source, cfg, condition)
        arrays.update({f"{condition['id']}__{k}": v for k, v in extra.items()})
        print(
            f"{condition['id']}: analysed in {time.perf_counter() - tick:.1f}s",
            flush=True,
        )
    with stage_run(
        ROOT,
        recipe.SLUG,
        "analyse",
        inputs={"compute": source},
        configuration={
            "isi_edges_ms": ISI_EDGES_MS.tolist(),
            "rhythmicity": "exp054 contrast via snnlab.analysis.rhythmicity_scalars",
            "cycles": "I-volley peaks (snnlab detect_population_bursts, peak-to-peak cycles)",
            "spectrum": "Welch, 256-sample Hann segments of the 1-ms population rate",
            "blocks": f"{cfg['block_presentations']} consecutive presentations",
            "mean_pixel": cfg["mean_pixel"],
        },
    ) as run:
        np.savez_compressed(run.export / "series.npz", **arrays)
        write_json_atomic(
            run.export / "numbers.json",
            dict(
                schema=f"{recipe.SLUG}.analysis/v1",
                configuration=cfg,
                rules=rules["rules"],
                scaling_targets_hz=rules["scaling_targets_hz"],
                isi_edges_ms=ISI_EDGES_MS.tolist(),
                results=results,
                summary=summary(results, cfg),
                compute_seconds=source.details["execution"]["elapsed_s"],
                condition_seconds=source.details["execution"]["condition_seconds"],
            ),
        )
        run.details["execution"]["elapsed_s"] = time.perf_counter() - started
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    analyse(parser.parse_args().source)
