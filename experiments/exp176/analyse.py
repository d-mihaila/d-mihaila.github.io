"""Measure rates, ISIs, population-rhythm spectra, weights and accuracy from the exp176 stream."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
from experiments.exp176 import recipe
from pingstore.library import source_run, stage_run, write_json_atomic
from snnlab import analysis

ISI_EDGES_MS = np.logspace(0, np.log10(2000), 41)
# Eoin's PING rhythmicity metric (exp054): population spike autocorrelogram
# up to 100-ms lag in 1-ms bins; the first 100 ms of the stream are skipped.
RHYTHMICITY = dict(max_lag_ms=100.0, bin_ms=1.0, burn_ms=100.0)


def side_peak_lag(lags, ac, trough_lag):
    """Lag of the autocorrelogram's first local maximum after the trough (the period).

    Uses the same 3-point smoothing as the contrast. None unless the side peak
    rises above the independence baseline of 1 and stands clearly above the
    trough, (peak − trough)/(peak + trough) >= 0.1, so noise is not a period.
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


def rhythmicity(raster, dt):
    """exp054 contrast (lobe − trough)/(lobe + trough), lags and autocorrelogram."""
    lag, width = RHYTHMICITY["max_lag_ms"], RHYTHMICITY["bin_ms"]
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
            iei_mode_ms=scalars["iei_mode_lag"],
            rhythm_period_ms=period,
            rhythm_frequency_hz=None if period is None else 1000.0 / period,
        ),
        lags,
        ac,
    )


def dense_raster(times_ms, cells, start_ms, stop_ms, dt_ms, n_cells):
    """Binary (time, cell) raster for one block of presentations."""
    steps = int(round((stop_ms - start_ms) / dt_ms))
    raster = np.zeros((steps, n_cells), dtype=np.uint8)
    keep = (times_ms >= start_ms) & (times_ms < stop_ms)
    index = np.rint((times_ms[keep] - start_ms) / dt_ms).astype(np.int64)
    raster[np.clip(index, 0, steps - 1), cells[keep].astype(np.int64)] = 1
    return raster


def block_measures(raster, cfg):
    """ISI and spectral measures of one population in one block, via snnlab.analysis."""
    dt = cfg["dt_ms"]
    isi = analysis.inter_spike_intervals(raster, dt)
    pooled = np.concatenate(isi["intervals_ms"]) if isi["intervals_ms"] else np.empty(0)
    cv = analysis.isi_cv(raster, dt, aggregation="mean")
    trace = analysis.population_rate_trace(raster, dt, bin_ms=cfg["rate_bin_ms"])
    rate = trace["rate_hz"]
    spectrum = analysis.power_spectrum(
        rate, cfg["rate_bin_ms"], method="welch", nperseg=256, detrend="constant"
    )
    band = cfg["rhythm_band_hz"]
    if spectrum["status"] == "ok":
        peak = analysis.spectral_peak(
            spectrum["frequencies_hz"], spectrum["power"], band
        )
        power = analysis.band_power(spectrum["frequencies_hz"], spectrum["power"], band)
        total = analysis.band_power(
            spectrum["frequencies_hz"],
            spectrum["power"],
            [1.0, 1000 / (2 * cfg["rate_bin_ms"])],
        )
        rhythm_fraction = (
            power["power"] / total["power"]
            if power["power"] is not None and total["power"]
            else None
        )
    else:
        peak, power, rhythm_fraction = (
            {"frequency_hz": None, "peak_power": None},
            {"power": None},
            None,
        )
    return dict(
        rate_hz=float(raster.sum() / raster.shape[1] / (raster.shape[0] * dt / 1000)),
        active_fraction=float((raster.sum(axis=0) > 0).mean()),
        isi_count=int(len(pooled)),
        isi_median_ms=float(np.median(pooled)) if len(pooled) else None,
        isi_cv_mean=cv["aggregate"],
        isi_histogram=np.histogram(pooled, bins=ISI_EDGES_MS)[0].tolist(),
        rhythm_peak_hz=peak["frequency_hz"],
        rhythm_peak_power=peak["peak_power"],
        rhythm_band_power=power["power"],
        rhythm_fraction=rhythm_fraction,
        spectrum_status=spectrum["status"],
    ), spectrum


def analyse_condition(source, cfg, network, condition):
    pops = recipe.populations(network)
    sizes = {"E": cfg["n_e"], "I": cfg["n_i"]}
    block = cfg["isi_block_presentations"]
    blocks = cfg["presentations"] // block
    seconds = cfg["presentation_ms"] / 1000
    with np.load(
        source.file(network, condition, "spikes.npz"), allow_pickle=False
    ) as spikes:
        data = {key: spikes[key] for key in spikes.files}
    result, spectra = {"rates_hz": {}, "blocks": {}}, {}
    for p in pops:
        counts = data[p + "_counts"]
        if len(counts) != cfg["presentations"] or counts.sum() != len(
            data[p + "_cell"]
        ):
            raise ValueError("Spike evidence does not match its counts")
        result["rates_hz"][p] = (counts / sizes[p] / seconds).tolist()
        rows, powers, curves = [], [], []
        for b in range(blocks):
            start = b * block * cfg["presentation_ms"]
            stop = (b + 1) * block * cfg["presentation_ms"]
            raster = dense_raster(
                data[p + "_time_ms"],
                data[p + "_cell"],
                start,
                stop,
                cfg["dt_ms"],
                sizes[p],
            )
            measures, spectrum = block_measures(raster, cfg)
            burn = int(round(RHYTHMICITY["burn_ms"] / cfg["dt_ms"])) if b == 0 else 0
            rhythm, lags, ac = rhythmicity(raster[burn:], cfg["dt_ms"])
            measures.update(rhythm, start_ms=start, stop_ms=stop)
            rows.append(measures)
            curves.append(np.asarray(ac, dtype=float))
            powers.append(np.asarray(spectrum["power"], dtype=float))
            frequencies = np.asarray(spectrum["frequencies_hz"], dtype=float)
        result["blocks"][p] = rows
        spectra[p] = dict(
            frequencies_hz=frequencies,
            power=np.stack(powers),
            autocorrelogram_lags_ms=np.asarray(lags, dtype=float),
            autocorrelogram=np.stack(curves),
        )
    with np.load(
        source.file(network, condition, "accuracy.npz"), allow_pickle=False
    ) as values:
        boundaries, correct = values["boundaries"], values["correct"]
    if correct.ndim != 2 or len(boundaries) != len(correct) or boundaries[0] != 0:
        raise ValueError("Accuracy evidence must start at the trained weights")
    images = correct.shape[1]
    curve = [float(100 * row.mean()) if images else None for row in correct]
    # Frozen weights never change, so its single evaluation holds at every boundary.
    result["accuracy"] = dict(
        boundaries=boundaries.tolist(),
        percent_curve=curve,
        correct=correct.sum(axis=1).tolist(),
        images=int(images),
        percent=curve[-1],
    )
    arrays = {}
    if condition == "van_rossum":
        with np.load(
            source.file(network, condition, "weights.npz"), allow_pickle=False
        ) as w:
            weights = {key: w[key] for key in w.files}
        result["weights"] = dict(
            mean_us=weights["mean"].tolist(),
            std_us=weights["std"].tolist(),
            quantiles_us=weights["quantiles"].tolist(),
            mean_abs_change_us=weights["mean_abs_change"].tolist(),
            net_change_us=weights["net_change"].tolist(),
            correlation_with_initial=weights["correlation_with_initial"].tolist(),
            scaling_rate_mean_hz=weights["scaling_rate_hz"].mean(axis=1).tolist(),
            afferent_sum_cv=(
                weights["afferent_sum"].std(axis=1)
                / weights["afferent_sum"].mean(axis=1)
            ).tolist(),
        )
        arrays.update(
            weight_histogram=weights["histogram"],
            weight_histogram_edges=weights["histogram_edges"],
            afferent_sum=weights["afferent_sum"],
            scaling_rate_hz=weights["scaling_rate_hz"],
            tracked_index=weights["tracked_index"],
            tracked_weights=weights["tracked_weights"],
        )
        with np.load(
            source.file(network, condition, "weight_snapshots.npz"), allow_pickle=False
        ) as snaps:
            boundaries = snaps["boundaries"]
            first = snaps[f"w{boundaries[0]:04d}"]
            last = snaps[f"w{boundaries[-1]:04d}"]
        chosen = np.argsort(-np.linalg.norm(first, axis=0))[: cfg["receptive_fields"]]
        arrays.update(
            rf_neurons=chosen,
            rf_initial=first[:, chosen].T,
            rf_final=last[:, chosen].T,
            snapshot_boundaries=boundaries,
        )
        result["receptive_field_neurons"] = chosen.tolist()
    for p, values in spectra.items():
        arrays[p + "_spectrum_frequencies_hz"] = values["frequencies_hz"]
        arrays[p + "_spectrum_power"] = values["power"]
        arrays[p + "_autocorrelogram_lags_ms"] = values["autocorrelogram_lags_ms"]
        arrays[p + "_autocorrelogram"] = values["autocorrelogram"]
    return result, arrays


def summary(results, cfg):
    """Start-versus-end numbers quoted in the report."""
    rows = {}
    for network in recipe.NETWORKS:
        for condition in recipe.CONDITIONS:
            r = results[network][condition]
            e = r["blocks"]["E"]
            row = dict(
                e_rate_first_hz=e[0]["rate_hz"],
                e_rate_last_hz=e[-1]["rate_hz"],
                isi_cv_first=e[0]["isi_cv_mean"],
                isi_cv_last=e[-1]["isi_cv_mean"],
                isi_median_first_ms=e[0]["isi_median_ms"],
                isi_median_last_ms=e[-1]["isi_median_ms"],
                rhythm_peak_first_hz=e[0]["rhythm_peak_hz"],
                rhythm_peak_last_hz=e[-1]["rhythm_peak_hz"],
                rhythm_fraction_first=e[0]["rhythm_fraction"],
                rhythm_fraction_last=e[-1]["rhythm_fraction"],
                rhythm_contrast_first=e[0]["rhythm_contrast"],
                rhythm_contrast_last=e[-1]["rhythm_contrast"],
                rhythm_frequency_first_hz=e[0]["rhythm_frequency_hz"],
                rhythm_frequency_last_hz=e[-1]["rhythm_frequency_hz"],
                accuracy_percent=r["accuracy"]["percent"],
            )
            if "I" in r["blocks"]:
                row.update(
                    i_rate_first_hz=r["blocks"]["I"][0]["rate_hz"],
                    i_rate_last_hz=r["blocks"]["I"][-1]["rate_hz"],
                )
            if "weights" in r:
                w = r["weights"]
                row.update(
                    weight_mean_first_us=w["mean_us"][0],
                    weight_mean_last_us=w["mean_us"][-1],
                    weight_mean_abs_change_last_us=w["mean_abs_change_us"][-1],
                    weight_correlation_last=w["correlation_with_initial"][-1],
                )
            rows[f"{network}/{condition}"] = row
    return rows


def analyse(identity):
    started = time.perf_counter()
    source = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="compute"
    )
    cfg = source.details["execution"]["configuration"]
    rules = json.loads(source.file("rules.json").read_text())
    results, arrays = {}, {}
    for network in recipe.NETWORKS:
        results[network] = {}
        for condition in recipe.CONDITIONS:
            tick = time.perf_counter()
            results[network][condition], extra = analyse_condition(
                source, cfg, network, condition
            )
            arrays.update({f"{network}__{condition}__{k}": v for k, v in extra.items()})
            print(
                f"{network}/{condition}: analysed in {time.perf_counter() - tick:.1f}s",
                flush=True,
            )
    with np.load(source.file("dataset.npz"), allow_pickle=False) as dataset:
        stream_labels = dataset["stream_labels"]
        arrays["stream_pixels_first"] = dataset["stream_pixels"][0]
        arrays["stream_pixels_last"] = dataset["stream_pixels"][-1]
    with stage_run(
        ROOT,
        recipe.SLUG,
        "analyse",
        inputs={"compute": source},
        configuration={
            "isi_edges_ms": ISI_EDGES_MS.tolist(),
            "spectrum": "Welch, 256-sample Hann segments of the 1-ms E/I population rate",
            "rhythmicity": dict(
                RHYTHMICITY,
                source="exp054 / snnlab.analysis.rhythmicity_scalars contrast",
            ),
            "isi_blocks": f"{cfg['isi_block_presentations']} consecutive presentations per block",
        },
    ) as run:
        np.savez_compressed(run.export / "series.npz", **arrays)
        write_json_atomic(
            run.export / "numbers.json",
            dict(
                schema="exp176.analysis/v1",
                configuration=cfg,
                rules=rules["rules"],
                scaling_target_rate_hz=rules["scaling_target_rate_hz"],
                isi_edges_ms=ISI_EDGES_MS.tolist(),
                rhythmicity_settings=RHYTHMICITY,
                stream_labels=stream_labels.tolist(),
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
