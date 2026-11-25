"""Render exp177 diagrams, rasters, rhythm, activity, weights, decoding and summary figures."""

import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp177 import recipe
from matplotlib.font_manager import FontProperties
from pingstore.library import source_run, stage_run
from snnlab import lang, viz

FONT_SCALE = 1.4
# Colour = plasticity rule. Frozen is the black control; the full van Rossum
# model (STDP + scaling) is the theme blue.
RULE_COLOR = {
    "frozen": theme.INK_BLACK,
    "stdp": theme.DEEP_RED,
    "scaling": theme.AMBER,
    "both": theme.ELECTRIC_CYAN,
}
SMALL = theme.SIZE_LEGEND * FONT_SCALE * 0.8


def circuit(bundle, path, title):
    """House-style native circuit; the plastic sensory projection is marked."""
    native = lang.diagram(bundle, view="circuit")
    symbols = {"drive": "X", "E": "E", "I": "I", "readout": "R"}
    diagram = replace(
        native,
        title=title,
        groups=(),
        nodes=tuple(node for node in native.nodes if node.id in symbols),
        edges=tuple(
            edge
            for edge in native.edges
            if edge.source in symbols and edge.target in symbols
        ),
    )
    positions = {"drive": (-460, 0), "E": (0, 0), "I": (0, -320), "readout": (400, 0)}
    ports = {
        "upper_left": (0.3, 0),
        "upper_right": (0.7, 0),
        "lower_left": (0.3, 1),
        "lower_right": (0.7, 1),
    }
    routes = {
        "input_to_E": viz.EdgeLayout(source_port="right", target_port="left"),
        "readout_projection": viz.EdgeLayout(source_port="right", target_port="left"),
        "E_to_I": viz.EdgeLayout(source_port="upper_left", target_port="lower_left"),
        "I_to_E": viz.EdgeLayout(source_port="lower_right", target_port="upper_right"),
    }
    texts = {"input_to_E": r"$W_{E,X}(t)$"}
    present = {node.id for node in diagram.nodes}
    layout = viz.DiagramLayout(
        nodes={
            key: viz.NodeLayout(
                position=position,
                label=f"${symbols[key]}$",
                min_size=(220, 150),
                ports=ports,
            )
            for key, position in positions.items()
            if key in present
        },
        edges={
            edge.id: replace(
                routes[edge.id],
                label=viz.DiagramLabel(
                    texts.get(
                        edge.id,
                        rf"$W_{{{symbols[edge.target]},{symbols[edge.source]}}}$",
                    )
                ),
            )
            for edge in diagram.edges
        },
        defaults=viz.LayoutDefaults(
            node=viz.NodeLayout(content_align="center", padding=16), gap=30
        ),
    )
    diagram_theme = viz.DiagramTheme(
        font_name=FontProperties(family="monospace").get_name(),
        background=theme.PAPER,
        ink=theme.INK_BLACK,
        muted=theme.MUTED,
        line=theme.RULE,
        neutral=theme.PAPER,
        signal=theme.INK_BLACK,
        inhibitory=theme.DEEP_RED,
        input_fill=theme.PAPER,
        input_border=theme.INK_BLACK,
        population_fill=theme.PAPER,
        population_border=theme.INK_BLACK,
        output=theme.PAPER,
        output_border=theme.INK_BLACK,
        title_size=theme.SIZE_PANEL * FONT_SCALE,
        label_size=theme.SIZE_TITLE * FONT_SCALE,
        secondary_size=theme.SIZE_LABEL * FONT_SCALE,
    )
    resolved = viz.resolve_diagram(diagram, layout=layout, theme=diagram_theme)
    for target in (path, path.with_suffix(".png")):
        viz.render_diagram(resolved, target)


def save(fig, export, name):
    for suffix in ("svg", "png"):
        fig.savefig(export / f"{name}.{suffix}", dpi=plt.rcParams["savefig.dpi"])
    plt.close(fig)


def tidy(axes):
    flat = list(np.atleast_1d(axes).ravel())
    for ax in flat:
        ax.spines[["top", "right"]].set_visible(False)
    for letter in theme.label_panels(flat):
        letter.set_fontsize(theme.SIZE_PANEL * FONT_SCALE)


def legend(ax):
    """Legend only when the panel has labelled lines."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=SMALL)


def values(rows, key):
    return np.array([np.nan if r.get(key) is None else r[key] for r in rows], float)


# ── Experiment hooks: panel grouping, colour and short label per condition ──


def panels(ctx):
    """One panel per network; every rule drawn in it."""
    groups = {}
    for c in ctx.cfg["conditions"]:
        groups.setdefault(c["network"], []).append(c)
    return [(recipe.NETWORK_LABELS[n], conditions) for n, conditions in groups.items()]


def color(ctx, c):
    return RULE_COLOR[c["rule"]]


def short(c):
    return recipe.RULE_LABELS[c["rule"]]


# ── Shared drawing helpers ──


def raster(ax, spikes, ctx, architecture, presentation, title):
    """I (red) above E (black), dots; one presentation."""
    cfg = ctx.cfg
    start = presentation * cfg["presentation_ms"]
    stop = start + cfg["presentation_ms"]
    offset = {"E": 0, "I": cfg["n_e"]}
    for p, ink in (("E", theme.INK_BLACK), ("I", theme.DEEP_RED)):
        if p not in recipe.populations(architecture):
            continue
        t, c = spikes[p + "_time_ms"], spikes[p + "_cell"]
        keep = (t >= start) & (t < stop)
        ax.scatter(
            t[keep] - start,
            c[keep] + offset[p],
            s=0.6,
            c=ink,
            linewidths=0,
            rasterized=True,
        )
    top = cfg["n_e"] + (cfg["n_i"] if architecture == "ping" else 0)
    if architecture == "ping":
        ax.axhline(cfg["n_e"] - 0.5, color=theme.MUTED, lw=0.5)
    ax.set(
        xlim=(0, cfg["presentation_ms"]),
        ylim=(-0.5, top - 0.5),
        title=title,
        xlabel="Time in presentation (ms)",
        ylabel="Neuron (E below I)" if architecture == "ping" else "E neuron",
    )
    ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.8)


def block_times(ctx, cid):
    n = ctx.results[cid]["presentations"] // ctx.cfg["block_presentations"]
    width = ctx.cfg["block_presentations"] * ctx.cfg["presentation_ms"] / 1000
    return (np.arange(n) + 0.5) * width


def grid(count, columns=4, width=4.6, height=3.9):
    """Wrapped grid of panels; unused panels are hidden."""
    rows = int(np.ceil(count / columns))
    cols = min(columns, count)
    fig, axes = plt.subplots(
        rows, cols, figsize=(width * cols, height * rows), squeeze=False
    )
    flat = list(axes.ravel())
    for ax in flat[count:]:
        ax.set_visible(False)
    return fig, flat[:count]


def name(ctx, c):
    return f"{recipe.NETWORK_LABELS[c['network']]} · {short(c)}"


def figure_rasters(ctx):
    """Start of the stream, then the last presentation of every condition."""
    conditions = ctx.cfg["conditions"]
    starts = list({c["network"]: c for c in conditions}.values())
    fig, axes = grid(len(starts) + len(conditions))
    for ax, c in zip(axes, starts, strict=False):
        arch = ctx.cfg["networks"][c["network"]]["architecture"]
        raster(
            ax,
            ctx.spikes[c["id"]],
            ctx,
            arch,
            0,
            f"{recipe.NETWORK_LABELS[c['network']]} · start",
        )
    for ax, c in zip(axes[len(starts) :], conditions, strict=True):
        arch = ctx.cfg["networks"][c["network"]]["architecture"]
        last = ctx.results[c["id"]]["presentations"] - 1
        raster(ax, ctx.spikes[c["id"]], ctx, arch, last, f"{name(ctx, c)} · end")
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "rasters")


def time_course_figure(ctx, metrics, filename, *, population="E"):
    """Rows = metrics, columns = panels; one line per condition."""
    groups = panels(ctx)
    fig, axes = plt.subplots(
        len(metrics),
        len(groups),
        figsize=(7.0 * len(groups), 3.4 * len(metrics)),
        squeeze=False,
    )
    for col, (title, conditions) in enumerate(groups):
        for row, (key, ylabel, limits) in enumerate(metrics):
            ax = axes[row, col]
            for c in conditions:
                rows = ctx.results[c["id"]]["blocks"].get(population)
                if rows:
                    ax.plot(
                        block_times(ctx, c["id"]),
                        values(rows, key),
                        "o-",
                        color=color(ctx, c),
                        label=short(c),
                    )
            ax.set(xlabel="Stream time (s)", ylabel=ylabel, title=f"{title}: {ylabel}")
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.85)
            if limits:
                ax.set_ylim(*limits)
            legend(ax)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, filename)


def figure_rhythm(ctx):
    """Rhythmicity, frequency, cycle rate and per-cycle participation over time."""
    time_course_figure(
        ctx,
        (
            ("rhythm_contrast", "E rhythmicity contrast", (0, 1)),
            ("rhythm_frequency_hz", "Rhythm frequency (Hz)", None),
            ("cycle_rate_hz", "I volleys per s", None),
            ("participation", "E cells firing per cycle", None),
        ),
        "rhythm",
    )


def figure_activity(ctx):
    """E/I rates, ISI irregularity and typical interval per block."""
    time_course_figure(
        ctx,
        (
            ("rate_hz", "E rate (Hz)", None),
            ("isi_cv_mean", "Mean E ISI CV", None),
            ("isi_median_ms", "Median E ISI (ms)", None),
            ("spectral_peak_hz", "Spectral peak (Hz)", None),
        ),
        "activity",
    )


def figure_autocorrelograms(ctx):
    """E spike autocorrelograms, first block (faint) versus last block (strong)."""
    conditions = ctx.cfg["conditions"]
    fig, axes = grid(len(conditions))
    for ax, c in zip(axes, conditions, strict=True):
        lags = ctx.series[f"{c['id']}__E_autocorrelogram_lags_ms"]
        curves = ctx.series[f"{c['id']}__E_autocorrelogram"]
        for index, alpha, label in ((0, 0.35, "first block"), (-1, 1.0, "last block")):
            ax.plot(
                lags[1:],
                curves[index][1:],
                color=color(ctx, c),
                alpha=alpha,
                label=label,
            )
        ax.axhline(1, color=theme.MUTED, ls=":", lw=0.8)
        ax.set(
            xlim=(0, 50),
            xlabel="Lag (ms)",
            ylabel="Autocorrelation",
            title=name(ctx, c),
        )
        ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.8)
        legend(ax)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "autocorrelograms")


def figure_weights(ctx):
    """Plastic X→E matrix: mean, accumulated change, memory, skew and scaling sensor."""
    seconds = ctx.cfg["presentation_ms"] / 1000
    metrics = (
        ("mean_us", "Mean W / trained mean", True),
        ("mean_abs_change_us", "Mean |ΔW| / trained mean", True),
        ("correlation_with_initial", "r with trained W", False),
        ("skew", "Weight skewness", False),
        ("scaling_rate_mean_hz", "Scaling sensor (Hz)", False),
    )
    groups = [
        (title, [c for c in conditions if c["rule"] != "frozen"])
        for title, conditions in panels(ctx)
    ]
    groups = [g for g in groups if g[1]]
    fig, axes = plt.subplots(
        len(groups),
        len(metrics),
        figsize=(5.0 * len(metrics), 4.0 * len(groups)),
        squeeze=False,
    )
    for row, (title, conditions) in enumerate(groups):
        for col, (key, ylabel, relative) in enumerate(metrics):
            ax = axes[row, col]
            for c in conditions:
                w = ctx.results[c["id"]].get("weights")
                if not w or not w.get(key):
                    continue
                y = np.array(w[key], float)
                if relative:
                    y = y / w["mean_us"][0]
                start = 1 if key == "scaling_rate_mean_hz" else 0
                ax.plot(
                    (np.arange(len(y)) + start) * seconds,
                    y,
                    color=color(ctx, c),
                    label=short(c),
                )
            ax.set(xlabel="Stream time (s)", ylabel=ylabel, title=f"{title}: {ylabel}")
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.85)
            legend(ax)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "weights")


def figure_synapses(ctx):
    """Trajectories of the tracked random synapses under each plastic condition."""
    conditions = [c for c in ctx.cfg["conditions"] if c["rule"] != "frozen"]
    seconds = ctx.cfg["presentation_ms"] / 1000
    fig, axes = grid(len(conditions), width=5.0)
    for ax, c in zip(axes, conditions, strict=True):
        tracks = ctx.series[f"{c['id']}__tracked_weights"]
        t = np.arange(len(tracks)) * seconds
        order = np.argsort(tracks[0])
        for rank, synapse in enumerate(order):
            ax.plot(
                t,
                tracks[:, synapse],
                color=color(ctx, c),
                alpha=0.05 + 0.45 * (rank / max(len(order) - 1, 1)) ** 4,
                lw=0.5,
            )
        ax.axhline(
            ctx.numbers["rules"][c["id"]]["equilibrium_us"], color=theme.MUTED, ls=":"
        )
        ax.set(xlabel="Stream time (s)", ylabel="Weight (µS)", title=name(ctx, c))
        ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.8)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "synapses")


def figure_decoding(ctx):
    """Held-out test accuracy and online decoding of the stream digits."""
    seconds = ctx.cfg["presentation_ms"] / 1000
    groups = panels(ctx)
    fig, axes = plt.subplots(
        len(groups), 2, figsize=(15, 4.2 * len(groups)), squeeze=False
    )
    for row, (title, conditions) in enumerate(groups):
        for c in conditions:
            r = ctx.results[c["id"]]
            style = dict(color=color(ctx, c), label=short(c))
            acc = r["accuracy"]
            if c["rule"] == "frozen":
                axes[row, 0].axhline(acc["percent"], ls="--", **style)
            else:
                axes[row, 0].plot(
                    np.array(acc["boundaries"]) * seconds,
                    acc["percent_curve"],
                    "o-",
                    **style,
                )
            axes[row, 1].plot(
                block_times(ctx, c["id"]),
                100 * np.array(r["online_decoding"]),
                "o-",
                **style,
            )
        images = ctx.results[conditions[0]["id"]]["accuracy"]["images"]
        axes[row, 0].set(
            ylim=(0, 100),
            xlabel="Stream time (s)",
            ylabel="Test accuracy (%)",
            title=f"{title}: {images} held-out test images",
        )
        axes[row, 1].set(
            ylim=(0, 100),
            xlabel="Stream time (s)",
            ylabel="Stream digits decoded (%)",
            title=f"{title}: online decoding per block",
        )
        for ax in axes[row]:
            legend(ax)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "decoding")


SUMMARY_METRICS = (
    ("rhythm_contrast", "Δ rhythmicity contrast"),
    ("rhythm_frequency_hz", "Δ rhythm frequency (Hz)"),
    ("participation", "Δ E cells per cycle"),
    ("rate_hz", "Δ E rate (Hz)"),
    ("accuracy", "Δ test accuracy (points)"),
    ("weight_change", "Mean |ΔW| / trained mean"),
)


def change(ctx, c, key):
    """Last block minus first block (accuracy: end minus start; weights: end)."""
    s = ctx.numbers["summary"][c["id"]]
    if key == "accuracy":
        first, last = s["accuracy_first"], s["accuracy_last"]
    elif key == "weight_change":
        return s.get("weight_change_last", 0.0)
    else:
        first, last = s.get(f"{key}_first"), s.get(f"{key}_last")
    return np.nan if first is None or last is None else last - first


def figure_summary(ctx):
    """Change from the first to the last block: networks on x, rules as bars."""
    groups = panels(ctx)
    fig, axes = plt.subplots(
        1, len(SUMMARY_METRICS), figsize=(4.0 * len(SUMMARY_METRICS), 4.4)
    )
    for ax, (key, title) in zip(axes, SUMMARY_METRICS, strict=True):
        width = 0.8 / max(len(g[1]) for g in groups)
        for g, (_, conditions) in enumerate(groups):
            for k, c in enumerate(conditions):
                ax.bar(
                    g + (k - (len(conditions) - 1) / 2) * width,
                    change(ctx, c, key),
                    width=width,
                    color=color(ctx, c),
                    label=short(c) if g == 0 else None,
                )
        ax.axhline(0, color=theme.MUTED, lw=0.8)
        ax.set_xticks(range(len(groups)), [g[0] for g in groups])
        ax.set_title(title)
        ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.85)
    legend(axes[0])
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "summary")


FIGURES = (
    figure_rasters,
    figure_rhythm,
    figure_autocorrelograms,
    figure_activity,
    figure_weights,
    figure_synapses,
    figure_decoding,
    figure_summary,
)


def present(identity):
    started = time.perf_counter()
    source = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="analyse"
    )
    numbers = json.loads(source.file("numbers.json").read_text())
    cfg = numbers["configuration"]
    pin = source.details["inputs"]["compute"]
    compute = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        reference=pin,
        experiment=recipe.SLUG,
        stage="compute",
    )
    series = dict(np.load(source.file("series.npz"), allow_pickle=False))
    spikes = {
        c["id"]: dict(np.load(compute.file(c["id"], "spikes.npz"), allow_pickle=False))
        for c in cfg["conditions"]
    }
    theme.set_paper_mode(True)
    theme.apply()
    plt.rcParams.update(
        {
            "font.size": theme.SIZE_BASE * FONT_SCALE,
            "axes.titlesize": theme.SIZE_TITLE * FONT_SCALE,
            "axes.labelsize": theme.SIZE_LABEL * FONT_SCALE,
            "xtick.labelsize": theme.SIZE_TICK * FONT_SCALE,
            "ytick.labelsize": theme.SIZE_TICK * FONT_SCALE,
            "legend.fontsize": theme.SIZE_LEGEND * FONT_SCALE,
        }
    )
    with stage_run(
        ROOT,
        recipe.SLUG,
        "present",
        inputs={"analysis": source},
        configuration={
            "style": "experiments.theme paper; enlarged article typography",
            "font_scale": FONT_SCALE,
            "colours": "rule: frozen black, STDP red, scaling amber, STDP + scaling blue",
            "figures": [f.__name__.removeprefix("figure_") for f in FIGURES],
        },
    ) as run:
        shutil.copyfile(source.file("numbers.json"), run.export / "numbers.json")
        for network, spec in cfg["networks"].items():
            bundle_path = compute.restore_bundle(
                f"{network}/network.bundle", run.scratch / f"{network}.bundle"
            )
            circuit(
                lang.load_bundle(bundle_path),
                run.export / f"circuit-{network}.svg",
                f"{recipe.NETWORK_LABELS[network]} ({spec['cell']}), plastic X→E",
            )
        ctx = SimpleNamespace(
            export=run.export,
            numbers=numbers,
            cfg=cfg,
            results=numbers["results"],
            series=series,
            spikes=spikes,
        )
        for figure in FIGURES:
            figure(ctx)
        run.details["execution"]["elapsed_s"] = time.perf_counter() - started
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    present(parser.parse_args().source)
