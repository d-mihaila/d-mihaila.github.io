"""Render exp005 diagrams, rasters, rhythm, activity, weights, decoding and summary figures."""

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
from experiments.exp005 import recipe
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


# ── Experiment hooks: panel grouping, colour, line style and labels ──

STREAM_SHORT = {"biased": "biased", "rate_step": "rate step", "long": "long"}


def panels(ctx):
    """One panel per stream; every network and rule on that stream drawn in it."""
    groups = {}
    for c in ctx.cfg["conditions"]:
        groups.setdefault(c["stream"], []).append(c)
    out = []
    for stream, conditions in groups.items():
        networks = {c["network"] for c in conditions}
        suffix = (
            f" · {recipe.NETWORK_LABELS[c['network']]}" if len(networks) == 1 else ""
        )
        out.append((recipe.STREAM_LABELS[stream] + suffix, conditions))
    return out


def color(ctx, c):
    return RULE_COLOR[c["rule"]]


def dash(c):
    """PING solid, COBA dashed."""
    return "--" if c["network"] == "coba" else "-"


def short(c):
    rule = recipe.RULE_LABELS[c["rule"]]
    if c["stream"] == "long":
        return f"{recipe.NETWORK_LABELS[c['network']]} · {rule}"
    return rule


def mark(ctx, ax, conditions):
    """Dotted line where the stream changes (biased phase end, rate step)."""
    boundary = recipe.phase_boundary(ctx.cfg, conditions[0]["stream"])
    if boundary is not None:
        ax.axvline(
            boundary * ctx.cfg["presentation_ms"] / 1000,
            color=theme.MUTED,
            ls=":",
            lw=1.2,
        )


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
    return (
        f"{recipe.NETWORK_LABELS[c['network']]} · {STREAM_SHORT[c['stream']]} · "
        f"{recipe.RULE_LABELS[c['rule']]}"
    )


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
                        marker="o",
                        ls=dash(c),
                        color=color(ctx, c),
                        label=short(c),
                    )
            mark(ctx, ax, conditions)
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
                    ls=dash(c),
                    color=color(ctx, c),
                    label=short(c),
                )
            mark(ctx, ax, conditions)
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
        mark(ctx, ax, [c])
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
            line = dict(style, marker="o", ls=dash(c))
            acc = r["accuracy"]
            if c["rule"] == "frozen":
                axes[row, 0].axhline(acc["percent"], ls=dash(c), lw=1, **style)
            else:
                axes[row, 0].plot(
                    np.array(acc["boundaries"]) * seconds, acc["percent_curve"], **line
                )
            axes[row, 1].plot(
                block_times(ctx, c["id"]), 100 * np.array(r["online_decoding"]), **line
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
            mark(ctx, ax, conditions)
            legend(ax)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "decoding")


def of_stream(ctx, stream, *, plastic=False):
    return [
        c
        for c in ctx.cfg["conditions"]
        if c["stream"] == stream and not (plastic and c["rule"] == "frozen")
    ]


def finish(axes, ctx, conditions, titles):
    for ax, title in zip(axes, titles, strict=True):
        ax.set(xlabel="Stream time (s)", title=title)
        ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.85)
        mark(ctx, ax, conditions)
        legend(ax)


def figure_competition(ctx):
    """Biased stream: accuracy on the shown digit and the others; digit-pixel weights."""
    conditions = of_stream(ctx, "biased")
    seconds = ctx.cfg["presentation_ms"] / 1000
    digit = ctx.cfg["biased_digit"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
    for c in conditions:
        r, style = ctx.results[c["id"]], dict(color=color(ctx, c), label=short(c))
        acc = r["accuracy"]
        if "digit_curve" in acc:
            t = np.array(acc["boundaries"]) * seconds
            axes[0].plot(t, acc["digit_curve"], marker="o", **style)
            axes[1].plot(t, acc["other_curve"], marker="o", **style)
        w = r.get("weights")
        if w and w.get("digit_pixel_mean_us"):
            t = np.arange(len(w["digit_pixel_mean_us"])) * seconds
            d, o = (np.array(w[f"{k}_pixel_mean_us"]) for k in ("digit", "other"))
            axes[2].plot(t, d / d[0], **style)
            axes[2].plot(t, o / o[0], ls="--", color=style["color"])
    axes[0].set(ylim=(0, 100), ylabel=f"Test accuracy on digit {digit} (%)")
    axes[1].set(ylim=(0, 100), ylabel="Test accuracy on other digits (%)")
    axes[2].set(ylabel="Mean weight / trained (solid digit, dashed other pixels)")
    axes[2].axhline(1, color=theme.MUTED, lw=0.6)
    finish(
        axes,
        ctx,
        conditions,
        (
            f"Shown digit ({digit})",
            "Other digits",
            f"Weights from digit-{digit} pixels",
        ),
    )
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "competition")


def figure_rate_step(ctx):
    """Input-rate step: E rate per presentation, weight mean and rhythm frequency."""
    conditions = of_stream(ctx, "rate_step")
    seconds = ctx.cfg["presentation_ms"] / 1000
    network = conditions[0]["network"]
    target = ctx.numbers["scaling_targets_hz"][network].get("rate_step")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
    for c in conditions:
        r, style = ctx.results[c["id"]], dict(color=color(ctx, c), label=short(c))
        rates = np.array(r["rates_hz"]["E"])
        axes[0].plot((np.arange(len(rates)) + 0.5) * seconds, rates, **style)
        w = r.get("weights")
        if w:
            y = np.array(w["mean_us"]) / w["mean_us"][0]
            axes[1].plot(np.arange(len(y)) * seconds, y, **style)
        axes[2].plot(
            block_times(ctx, c["id"]),
            values(r["blocks"]["E"], "rhythm_frequency_hz"),
            marker="o",
            **style,
        )
    if target:
        axes[0].axhline(target, color=theme.MUTED, ls="--", lw=1, label="setpoint ν*")
    axes[0].set(ylabel="E rate per presentation (Hz)")
    axes[1].set(ylabel="Mean W / trained mean")
    axes[2].set(ylabel="Rhythm frequency (Hz)")
    factor = ctx.cfg["rate_step_factor"]
    finish(
        axes,
        ctx,
        conditions,
        (f"E rate; input ×{factor:g} at the line", "Sensory weights", "PING rhythm"),
    )
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "rate_step")


def figure_relaxation(ctx):
    """Long stream: memory of the trained weights, drift, skew and final histogram."""
    conditions = of_stream(ctx, "long", plastic=True)
    seconds = ctx.cfg["presentation_ms"] / 1000
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.2))
    for c in conditions:
        r, w = ctx.results[c["id"]], ctx.results[c["id"]]["weights"]
        style = dict(color=color(ctx, c), ls=dash(c))
        t = np.arange(len(w["mean_us"])) * seconds
        curves = (
            (axes[0], np.array(w["correlation_with_initial"]), "correlation"),
            (axes[1], np.array(w["mean_abs_change_us"]) / w["mean_us"][0], "change"),
        )
        for ax, y, key in curves:
            fit = r["relaxation"].get(key)
            label = short(c)
            if fit:
                label += f" (τ {fit['tau_s']:.0f} s" + (
                    ")" if fit["within_stream"] else ", > stream)"
                )
                y0, inf, tau = fit["start"], fit["asymptote"], fit["tau_s"]
                ax.plot(
                    t,
                    inf + (y0 - inf) * np.exp(-t / tau),
                    color=theme.MUTED,
                    lw=0.8,
                    ls=":",
                )
            ax.plot(t, y, label=label, **style)
        axes[2].plot(t, w["skew"], label=short(c), **style)
        hist = ctx.series[f"{c['id']}__weight_histogram"]
        edges = ctx.series[f"{c['id']}__weight_histogram_edges"]
        centres = (edges[:-1] + edges[1:]) / 2
        if c is conditions[0]:
            axes[3].plot(
                centres, hist[0] / hist[0].sum(), color=theme.MUTED, label="trained"
            )
        axes[3].plot(
            centres, hist[-1] / hist[-1].sum(), label=f"{short(c)}, end", **style
        )
    axes[0].set(ylabel="r with trained W")
    axes[1].set(ylabel="Mean |ΔW| / trained mean")
    axes[2].set(ylabel="Weight skewness")
    finish(
        axes[:3],
        ctx,
        conditions,
        ("Memory of the trained weights", "Accumulated change", "Distribution shape"),
    )
    axes[3].set(
        xlabel="Weight (µS)",
        ylabel="Fraction of synapses",
        yscale="log",
        title="Weight distribution",
    )
    axes[3].title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.85)
    legend(axes[3])
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "relaxation")


FIGURES = (
    figure_rasters,
    figure_rhythm,
    figure_autocorrelograms,
    figure_activity,
    figure_weights,
    figure_synapses,
    figure_decoding,
    figure_competition,
    figure_rate_step,
    figure_relaxation,
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
        experiment=recipe.source_experiment(pin["run_id"]),
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
            "colours": "rule: frozen black, STDP red, STDP + scaling blue; COBA dashed",
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
