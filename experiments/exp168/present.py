"""Render two-panel rasters for both recurrence cases."""

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments import theme
from experiments.exp168 import recipe
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from snnlab import lang, viz


def network_diagram(bundle, recurrence):
    """Resolve the saved circuit with explicit routes and target-first weight labels."""
    import matplotlib

    native = lang.diagram(bundle, view="circuit")
    symbols = {"X": "X", "E": "E", "I": "I"}
    diagram = replace(
        native,
        title="EE/II recurrence enabled"
        if recurrence == "on"
        else "EE/II projections absent",
        nodes=tuple(
            replace(node, detail=node.detail.splitlines()[0])
            for node in native.nodes
            if node.id in symbols
        ),
        edges=tuple(edge for edge in native.edges if edge.role != "output"),
        groups=(),
    )
    positions = {"X": (-400, 150), "E": (0, 0), "I": (0, 300)}
    routes = dict(
        X_to_E=viz.EdgeLayout(source_port="upper_right", target_port="left"),
        X_to_I=viz.EdgeLayout(source_port="lower_right", target_port="left"),
        E_to_I=viz.EdgeLayout(source_port="bottom_left", target_port="top_left"),
        I_to_E=viz.EdgeLayout(source_port="top_right", target_port="bottom_right"),
        E_to_E=viz.EdgeLayout(
            source_port="top_right", target_port="right", waypoints=((180, -130),)
        ),
        I_to_I=viz.EdgeLayout(
            source_port="bottom_right", target_port="right", waypoints=((180, 430),)
        ),
    )
    ports = dict(
        upper_left=(0, 0.3),
        lower_left=(0, 0.7),
        upper_right=(1, 0.3),
        lower_right=(1, 0.7),
        top_left=(0.25, 0),
        top_right=(0.75, 0),
        bottom_left=(0.25, 1),
        bottom_right=(0.75, 1),
    )
    layout = viz.DiagramLayout(
        nodes={
            identifier: viz.NodeLayout(
                position=position,
                label=f"${symbols[identifier]}$",
                min_size=(240, 160),
                ports=ports,
            )
            for identifier, position in positions.items()
        },
        edges={
            edge.id: replace(
                routes[edge.id],
                label=viz.DiagramLabel(
                    rf"$W_{{{symbols[edge.target]},{symbols[edge.source]}}}$"
                ),
            )
            for edge in diagram.edges
        },
        defaults=viz.LayoutDefaults(
            node=viz.NodeLayout(content_align="center", padding=16), gap=30
        ),
    )
    with matplotlib.rc_context({"savefig.bbox": None}):
        return viz.resolve_diagram(
            diagram, layout=layout, theme=viz.DiagramTheme(secondary_size=13)
        )


def panel(ax, events_path, cfg, title):
    start, end = cfg["measurement_ms"]
    with np.load(events_path, allow_pickle=False) as events:
        for population, color in (("E", theme.INK_BLACK), ("I", theme.DEEP_RED)):
            times = events[population + "_time_ms"]
            cells = events[population + "_cell"]
            mask = (times >= start) & (times < end)
            ax.scatter(
                times[mask] - start,
                cells[mask] + (0 if population == "E" else cfg["n_e"]),
                s=0.2,
                c=color,
                linewidths=0,
                rasterized=True,
            )
    ax.axhline(cfg["n_e"] - 0.5, color=theme.MUTED, lw=0.5)
    ax.set(
        xlim=(0, end - start),
        ylim=(-0.5, cfg["n_e"] + cfg["n_i"] - 0.5),
        xlabel="Time after burn-in (ms)",
        ylabel="Neuron index",
        title=title,
    )
    ax.spines[["top", "right"]].set_visible(False)


def transition_figure(origin, data, directory):
    """Render the native continuous trajectory and its recorded input-gain control."""
    import matplotlib.pyplot as plt

    cfg = data["configuration"]
    control = cfg["input_i_gain"]
    duration = cfg["duration_ms"]
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
        layout="constrained",
    )
    panel(
        axes[0],
        recipe.transition_file(origin, "events.npz"),
        cfg,
        "Fixed wiring, EE/II active; external source rate fixed",
    )
    axes[0].set_xlabel("")
    with np.load(
        recipe.transition_file(origin, "events.npz"), allow_pickle=False
    ) as events:
        for population, color in (("E", theme.INK_BLACK), ("I", theme.DEEP_RED)):
            bins = np.arange(0, duration + 1, 1.0)
            counts, _ = np.histogram(events[population + "_time_ms"], bins)
            axes[1].plot(
                bins[:-1],
                counts / cfg["n_" + population.lower()] * 1000,
                color=color,
                lw=0.7,
                label=population,
            )
    times = [0, control["start_ms"], control["end_ms"], duration]
    axes[2].plot(
        times, [cfg["input_e_gain"]] * 4, color=theme.INK_BLACK, label="External→E"
    )
    axes[2].plot(
        times,
        [control["before"], control["before"], control["after"], control["after"]],
        color=theme.DEEP_RED,
        label="External→I",
    )
    axes[1].set_ylabel("Rate (Hz)")
    axes[2].set(ylabel="Input gain", xlabel="Time (ms)")
    for ax in axes:
        ax.axvspan(
            control["start_ms"], control["end_ms"], color=theme.GREY_LIGHT, alpha=0.15
        )
    for ax in axes[1:]:
        ax.legend(loc="upper right", ncol=2)
    theme.label_panels(axes)
    for extension in ("png", "svg", "pdf"):
        figure.savefig(directory / f"gain-transition.{extension}", dpi=180)
    plt.close(figure)


def present(source, *, base_present=None, run_id=None):
    started = perf_counter()
    analysed = source_run(
        REPO / ".pingstore", source, experiment=recipe.SLUG, stage="analyse"
    )
    pins = analysed.details["inputs"]
    origins = {
        recurrence: source_run(
            REPO / ".pingstore",
            pins[f"compute_{recurrence}"]["run_id"],
            reference=pins[f"compute_{recurrence}"],
        )
        for recurrence in recipe.RECURRENT_CASES
    }
    numbers = load_json(analysed.file("numbers.json"))
    inputs = {"analyse": analysed}
    base = None
    if base_present:
        base = source_run(
            REPO / ".pingstore", base_present, experiment=recipe.SLUG, stage="present"
        )
        baseline = load_json(base.file("numbers.json"))
        if any(baseline[key] != numbers[key] for key in ("configurations", "metrics")):
            raise ValueError(
                "existing figures and current analysis use different comparison evidence"
            )
        inputs["comparison_figures"] = base
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme.apply()
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs=inputs,
        configuration=analysed.details["execution"]["configuration"],
        run_id=run_id,
    ) as run:
        for recurrence, prefix in (
            ("on", "recurrence-on-rasters"),
            ("off", "recurrence-off-rasters"),
        ):
            if base:
                for extension in ("png", "svg", "pdf"):
                    shutil.copy2(
                        base.file(f"{prefix}.{extension}"),
                        run.export / f"{prefix}.{extension}",
                    )
                continue
            origin = origins[recurrence]
            configurations = recipe.case_configuration(origin, recurrence)
            figure, axes = plt.subplots(2, 1, figsize=(12, 8.5), layout="constrained")
            for index, state in enumerate(recipe.CONDITIONS):
                panel(
                    axes[index],
                    recipe.case_file(origin, recurrence, state, "events.npz"),
                    configurations[state],
                    f"{chr(65 + index)}  {state}",
                )
            for extension in ("png", "svg", "pdf"):
                figure.savefig(run.export / f"{prefix}.{extension}", dpi=180)
            plt.close(figure)
        for recurrence in recipe.RECURRENT_CASES:
            if base:
                for extension in ("svg", "png"):
                    name = f"recurrence-{recurrence}-circuit.{extension}"
                    shutil.copy2(base.file(name), run.export / name)
                continue
            origin = origins[recurrence]
            state = recipe.CONDITIONS[0]
            bundle = lang.load_bundle(
                recipe.case_file(origin, recurrence, state, "network.bundle")
            )
            circuit = network_diagram(bundle, recurrence)
            with matplotlib.rc_context({"savefig.bbox": None}):
                for extension in ("svg", "png"):
                    viz.render_diagram(
                        circuit,
                        run.export / f"recurrence-{recurrence}-circuit.{extension}",
                    )
        if "gain_transition" in numbers:
            transition_origin = source_run(
                REPO / ".pingstore", pins["transition_compute"]["run_id"]
            )
            transition_figure(transition_origin, numbers["gain_transition"], run.export)
        write_json_atomic(
            run.export / "numbers.json",
            {
                **numbers,
                "figure": "both-and-transition"
                if "gain_transition" in numbers
                else "both",
            },
        )
        run.details["execution"]["elapsed_s"] = perf_counter() - started
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--base-present")
    present(**vars(parser.parse_args()))
