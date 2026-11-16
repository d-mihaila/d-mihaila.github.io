"""Analyse the pinned EE/II-on and EE/II-off native runs."""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp168 import recipe
from pingstore.library import source_run, stage_run, write_json_atomic
from snnlab import analysis


def _origin(run_id, reference):
    if reference is None:
        return source_run(REPO / ".pingstore", run_id)
    return source_run(
        REPO / ".pingstore",
        run_id,
        reference={
            "run_id": run_id,
        },
    )


def raster_from_events(times, cells, cfg, population):
    start, end = cfg["measurement_ms"]
    selected = (times >= start) & (times < end)
    matrix = np.zeros(
        (round((end - start) / cfg["dt_ms"]), cfg[f"n_{population.lower()}"]),
        dtype=bool,
    )
    steps = np.rint((times[selected] - start) / cfg["dt_ms"]).astype(np.int64)
    indices = cells[selected].astype(np.int64)
    if (
        np.any(steps < 0)
        or np.any(steps >= len(matrix))
        or np.any(indices < 0)
        or np.any(indices >= matrix.shape[1])
    ):
        raise ValueError("spike event outside recorded dimensions")
    matrix[steps, indices] = True
    if int(matrix.sum()) != int(selected.sum()):
        raise ValueError("duplicate spike events")
    return matrix, selected


def measure(path, cfg, *, transition=False):
    start, end = cfg["measurement_ms"]
    duration_s = (end - start) / 1000
    out = {}
    samples = []
    with np.load(path, allow_pickle=False) as events:
        for population in ("E", "I"):
            times = events[population + "_time_ms"]
            cells = events[population + "_cell"]
            raster, mask = raster_from_events(times, cells, cfg, population)
            native = analysis.rhythmicity_metrics(
                raster, cfg["dt_ms"], max_lag_ms=100.0, bin_ms=1.0
            )
            cv = analysis.isi_cv(
                raster, cfg["dt_ms"], min_spikes=20, ddof=0, aggregation="median"
            )
            size = cfg[f"n_{population.lower()}"]
            out[population + "_rate_hz"] = int(mask.sum()) / size / duration_s
            out[population + "_rhythmicity"] = native["contrast"]
            out[population + "_median_cv"] = cv["aggregate"]
            out[population + "_cv_cells"] = int(cv["valid"].sum())
            if transition:
                from scipy.signal import periodogram

                cv5 = analysis.isi_cv(
                    raster, cfg["dt_ms"], min_spikes=5, ddof=0, aggregation="median"
                )
                out[population + "_median_cv_min5"] = cv5["aggregate"]
                out[population + "_cv_min5_cells"] = int(cv5["valid"].sum())
                counts = raster.reshape(
                    round(end - start), round(1 / cfg["dt_ms"]), size
                ).sum(axis=(1, 2))
                frequencies, power = periodogram(
                    counts, fs=1000.0, window="hann", detrend="constant"
                )
                band = (frequencies >= 20) & (frequencies <= 100)
                out[population + "_gamma_peak_hz"] = float(
                    frequencies[band][np.argmax(power[band])]
                )
            width = round(1.0 / cfg["dt_ms"])
            samples.append(
                raster[:, :200].reshape(round(end - start), width, 200).any(axis=1)
            )
            del raster
    correlation = analysis.spike_count_correlations(
        np.concatenate(samples, axis=1).astype(float),
        1.0,
        bin_ms=20.0,
        aggregation="mean",
    )
    out["mean_correlation"] = correlation["aggregate"]
    return {k: float(v) if isinstance(v, np.floating) else v for k, v in out.items()}


def analyse(*, transition_source=None, run_id=None):
    started = perf_counter()
    origins = {
        recurrence: _origin(recipe.SOURCES[recurrence]["run_id"], None)
        for recurrence in recipe.RECURRENT_CASES
    }
    configurations = {}
    metrics = {}
    for recurrence, origin in origins.items():
        source_configurations = recipe.case_configuration(origin, recurrence)
        configurations[recurrence] = source_configurations
        metrics[recurrence] = {
            state: measure(
                recipe.case_file(origin, recurrence, state, "events.npz"),
                source_configurations[state],
            )
            for state in recipe.CONDITIONS
        }
    inputs = {"compute_on": origins["on"], "compute_off": origins["off"]}
    transition_data = None
    if transition_source:
        transition_origin = source_run(
            REPO / ".pingstore",
            transition_source,
            experiment=recipe.SLUG,
            stage="compute",
        )
        inputs["transition_compute"] = transition_origin
        cfg = transition_origin.details["execution"]["configuration"]["transition"]
        transition_data = dict(
            configuration=cfg,
            metrics={
                window: measure(
                    recipe.transition_file(transition_origin, "events.npz"),
                    {**cfg, "measurement_ms": bounds},
                    transition=True,
                )
                for window, bounds in cfg["windows"].items()
            },
        )
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs=inputs,
        configuration={
            "method": "snnlab.analysis.rhythmicity_metrics",
            "bin_ms": 1.0,
            "max_lag_ms": 100.0,
            "min_spikes_for_cv": 20,
            "correlation_cells_per_population": 200,
            "correlation_bin_ms": 20.0,
        },
        run_id=run_id,
    ) as run:
        write_json_atomic(
            run.export / "numbers.json",
            dict(
                schema="exp168.analysis.both/v1",
                configurations=configurations,
                metrics=metrics,
                **({"gain_transition": transition_data} if transition_data else {}),
                analysis={
                    "bin_ms": 1.0,
                    "max_lag_ms": 100.0,
                    "min_spikes_for_cv": 20,
                    "correlation_cells_per_population": 200,
                    "correlation_bin_ms": 20.0,
                },
                snnlab_version=version("snnlab"),
                evidence={"kind": "native"},
                limitations=[
                    "Each condition is one selected network/input realization.",
                    "AI and PING differ in three external-drive settings.",
                ],
            ),
        )
        run.details["execution"]["elapsed_s"] = perf_counter() - started
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--transition-source")
    analyse(**vars(parser.parse_args()))
