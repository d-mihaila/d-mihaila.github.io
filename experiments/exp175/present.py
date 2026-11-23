"""Plot retained learning curves, without making an independent-test claim."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]

from experiments import theme
from experiments.exp175 import recipe
from pingstore.library import source_run, stage_run, write_json_atomic


def present(identity):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    source = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="analyse"
    )
    summary = json.loads(source.file("summary.json").read_text())
    rows = summary["curve"]
    theme.apply()
    with stage_run(ROOT, recipe.SLUG, "present", inputs={"analyse": source}) as run:
        fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
        ax.axhline(
            summary["initial_validation_mse"],
            color=theme.INK_BLACK,
            linestyle="--",
            label="Initial encoder and fresh decoder",
        )
        ax.plot(
            [r["epoch"] for r in rows],
            [r["validation_loss"] for r in rows],
            color=theme.DEEP_RED,
            label="Joint training",
        )
        ax.set(xlabel="Epoch", ylabel="Validation pixel MSE (dimensionless)")
        ax.legend()
        fig.savefig(run.export / "learning-curves.svg")
        fig.savefig(run.export / "learning-curves.png", dpi=180)
        plt.close(fig)
        with np.load(source.file("reconstructions.npz")) as images:
            count = len(images["targets"])
            fig, axes = plt.subplots(
                count, 4, figsize=(8, 2 * count), squeeze=False, layout="constrained"
            )
            keys = ("images_A", "images_B", "targets", "reconstructed")
            for row in range(count):
                for col, key in enumerate(keys):
                    axes[row, col].imshow(
                        images[key][row].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1
                    )
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                axes[row, 0].set_ylabel(
                    f"Cue {'AB'[images['cue'][row]]} · phase {images['phase'][row] + 1}"
                )
            for ax, title in zip(
                axes[0],
                ("Image A", "Image B", "Attended target", "C reconstruction"),
                strict=True,
            ):
                ax.set_title(title)
            fig.savefig(run.export / "reconstructions.png", dpi=150)
            fig.savefig(run.export / "reconstructions.svg")
            plt.close(fig)
        write_json_atomic(run.export / "summary.json", summary)
        write_json_atomic(
            run.export / "timing.json",
            dict(total_seconds=time.perf_counter() - started),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    present(parser.parse_args().source)
