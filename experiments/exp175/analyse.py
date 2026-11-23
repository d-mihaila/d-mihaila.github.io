"""Summarise retained native validation history; no inference or new fitting."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]

from experiments.exp175 import recipe
from pingstore.library import source_run, stage_run, write_json_atomic


def summarize(metrics, checks):
    rows = metrics["epochs"]
    if not rows or rows[0]["epoch"] != 0:
        raise ValueError("Missing initial frozen-weight validation baseline")
    for row in rows:
        if not math.isfinite(row["validation_loss"]):
            raise ValueError("Incomplete native validation evidence")
    selected = next(row for row in rows if row["epoch"] == checks["selected_epoch"])
    if not checks["all_frozen_weights_identical"] or not checks["shared_encoder"]:
        raise ValueError("Training violated the frozen-body/shared-encoder contract")
    return dict(
        initial_validation_mse=rows[0]["validation_loss"],
        selected_validation_mse=selected["validation_loss"],
        final_validation_mse=rows[-1]["validation_loss"],
        selected_epoch=selected["epoch"],
        curve=rows,
        limitation=checks["limitation"],
        official_test_evaluated=False,
        attention_content_selection_verified=False,
    )


def analyse(identity):
    started = time.perf_counter()
    source = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="compute"
    )
    summary = summarize(
        json.loads(source.file("metrics.json").read_text()),
        json.loads(source.file("checks.json").read_text()),
    )
    with stage_run(ROOT, recipe.SLUG, "analyse", inputs={"compute": source}) as run:
        with np.load(source.file("reconstructions.npz")) as examples:
            np.savez_compressed(run.export / "reconstructions.npz", **dict(examples))
        write_json_atomic(run.export / "summary.json", summary)
        write_json_atomic(
            run.export / "timing.json",
            dict(total_seconds=time.perf_counter() - started),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    analyse(parser.parse_args().source)
