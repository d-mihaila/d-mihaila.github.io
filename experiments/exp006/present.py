"""Render exp006 diagrams, stimuli, rasters, rhythm, ISI, spectra and weight-change figures."""

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
from experiments.exp006 import recipe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from pingstore.library import source_run, stage_run
from snnlab import lang, viz

FONT_SCALE = 1.4
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


# ── Experiment hooks: panels by network; colour = rule (hue) and digit (shade) ──

RAMPS = {
    "frozen": LinearSegmentedColormap.from_list("frozen", ["#a6a6a6", theme.INK_BLACK]),
    "both": LinearSegmentedColormap.from_list(
        "both", ["#8fd9ec", theme.ELECTRIC_CYAN, "#00475a"]
    ),
}


def panels(ctx):
    """One panel per network; every digit and rule drawn in it."""
    return [
        (
            recipe.NETWORK_LABELS[network],
            [c for c in ctx.cfg["conditions"] if c["network"] == network],
        )
        for network in ctx.cfg["source_cells"]
    ]


def shade(ctx, rule, digit):
    digits = ctx.cfg["digits"]
    x = 0.6 if len(digits) == 1 else digits.index(digit) / (len(digits) - 1)
    return RAMPS[rule](x)


def color(ctx, c):
    return shade(ctx, c["rule"], c["digit"])


def dash(c):
    """Frozen dashed, STDP + scaling solid."""
    return "--" if c["rule"] == "frozen" else "-"


def short(c):
    return f"digit {c['digit']} · {recipe.RULE_LABELS[c['rule']]}"


def cid(network, digit, rule):
    return f"{network}__{recipe.stream_kind(digit)}__{rule}"


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
        xlabel="Time in window (ms)",
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
    """Per network and digit: first window, then the last window frozen and plastic."""
    pairs = [(n, d) for n in ctx.cfg["source_cells"] for d in ctx.cfg["digits"]]
    fig, axes = plt.subplots(
        len(pairs), 3, figsize=(4.8 * 3, 3.7 * len(pairs)), squeeze=False
    )
    for row, (network, digit) in enumerate(pairs):
        arch = ctx.cfg["networks"][network]["architecture"]
        frozen, plastic = cid(network, digit, "frozen"), cid(network, digit, "both")
        label = f"{recipe.NETWORK_LABELS[network]} · digit {digit}"
        last = ctx.results[frozen]["presentations"] - 1
        seconds = (last + 1) * ctx.cfg["presentation_ms"] / 1000
        raster(
            axes[row, 0], ctx.spikes[frozen], ctx, arch, 0, f"{label} · first window"
        )
        raster(
            axes[row, 1],
            ctx.spikes[frozen],
            ctx,
            arch,
            last,
            f"{label} · frozen, at {seconds:g} s",
        )
        raster(
            axes[row, 2],
            ctx.spikes[plastic],
            ctx,
            arch,
            last,
            f"{label} · STDP + scaling, at {seconds:g} s",
        )
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
            ax.set(xlabel="Time (s)", ylabel=ylabel, title=f"{title}: {ylabel}")
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
            ax.set(xlabel="Time (s)", ylabel=ylabel, title=f"{title}: {ylabel}")
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.85)
            legend(ax)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "weights")


def condition(network, digit, rule):
    """A condition record for colour and style lookups."""
    return dict(id=cid(network, digit, rule), network=network, digit=digit, rule=rule)


def fixed_point(ctx, key):
    """The STDP fixed point (the trained mean weight) as a fraction of w_max."""
    rule = ctx.numbers["rules"][key]
    return rule["equilibrium_us"] / rule["w_max_us"]


def style_legend(ax, solid, dashed):
    """Colour legend for the digits plus black proxies for the two line styles."""
    handles, labels = ax.get_legend_handles_labels()
    proxies = [
        Line2D([], [], color=theme.INK_BLACK, ls="-", label=solid),
        Line2D([], [], color=theme.INK_BLACK, ls="--", label=dashed),
    ]
    ax.legend(
        handles + proxies,
        labels + [solid, dashed],
        frameon=False,
        fontsize=SMALL,
    )


def binned(ctx, key, group, minimum=20):
    """Tendency curves with sparsely filled initial-weight bins removed."""
    curves = np.array(ctx.series[f"{key}__tendency_{group}"], float)
    counts = ctx.series[f"{key}__tendency_counts_{group}"]
    curves[:, counts < minimum] = np.nan
    edges = ctx.series[f"{key}__tendency_edges"]
    return (edges[:-1] + edges[1:]) / 2, curves


def network_digit_axes(ctx, width=5.2, height=4.2):
    networks, digits = list(ctx.cfg["source_cells"]), ctx.cfg["digits"]
    fig, axes = plt.subplots(
        len(networks),
        len(digits),
        figsize=(width * len(digits), height * len(networks)),
        squeeze=False,
    )
    return fig, axes, networks, digits


def figure_stimulus(ctx):
    """The images shown, with the digit pixels outlined."""
    data, threshold = ctx.stimulus, ctx.cfg["digit_pixel_threshold"]
    count = len(data["stimulus_images"])
    fig, axes = plt.subplots(1, count, figsize=(3.9 * count, 4.4), squeeze=False)
    for ax, image, digit, pixels in zip(
        axes[0],
        data["stimulus_images"],
        data["stimulus_digits"],
        data["stimulus_pixels"],
        strict=True,
    ):
        picture = np.asarray(pixels, float).reshape(28, 28)
        ax.imshow(picture, cmap="gray_r", vmin=0, vmax=1)
        ax.contour(
            (picture > threshold).astype(float),
            levels=[0.5],
            colors=[theme.ELECTRIC_CYAN],
            linewidths=1.6,
        )
        ax.set(
            xticks=[],
            yticks=[],
            title=f"Digit {int(digit)}: test image {int(image)}\n"
            f"{int((picture > threshold).sum())} digit pixels",
        )
        ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.8)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "stimulus")


def figure_synapses(ctx):
    """Tracked random synapses: digit-pixel synapses in colour, the others grey."""
    conditions = [c for c in ctx.cfg["conditions"] if c["rule"] != "frozen"]
    seconds = ctx.cfg["presentation_ms"] / 1000
    fig, axes = grid(len(conditions), columns=len(ctx.cfg["digits"]), width=5.0)
    for ax, c in zip(axes, conditions, strict=True):
        tracks = ctx.series[f"{c['id']}__tracked_weights"]
        index = ctx.series[f"{c['id']}__tracked_index"]
        mask = ctx.series[f"{c['id']}__digit_mask"]
        on = mask[index // ctx.cfg["n_e"]]
        t = np.arange(len(tracks)) * seconds
        for synapse in np.flatnonzero(~on):
            ax.plot(t, tracks[:, synapse], color=theme.MUTED, alpha=0.12, lw=0.5)
        for synapse in np.flatnonzero(on):
            ax.plot(t, tracks[:, synapse], color=color(ctx, c), alpha=0.55, lw=0.6)
        ax.axhline(
            ctx.numbers["rules"][c["id"]]["equilibrium_us"],
            color=theme.INK_BLACK,
            ls=":",
        )
        ax.set(
            xlabel="Time (s)",
            ylabel="Weight (µS)",
            title=f"{name(ctx, c)}\n{int(on.sum())} of {len(on)} on digit pixels",
        )
        ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.75)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "synapses")


def figure_isi(ctx):
    """E ISI distributions in the first and the last block, per network and digit."""
    edges = np.array(ctx.numbers["isi_edges_ms"])
    centres = np.sqrt(edges[:-1] * edges[1:])
    block_s = ctx.cfg["block_presentations"] * ctx.cfg["presentation_ms"] / 1000
    fig, axes, networks, digits = network_digit_axes(ctx)
    for row, network in enumerate(networks):
        for col, digit in enumerate(digits):
            ax = axes[row, col]
            for rule in recipe.DESIGN_RULES:
                c = condition(network, digit, rule)
                rows = ctx.results[c["id"]]["blocks"]["E"]
                for index, alpha, when in ((0, 0.4, "first"), (-1, 1.0, "last")):
                    counts = np.array(rows[index]["isi_histogram"], float)
                    if counts.sum():
                        ax.plot(
                            centres,
                            counts / counts.sum(),
                            ls=dash(c),
                            color=color(ctx, c),
                            alpha=alpha,
                            label=f"{recipe.RULE_LABELS[rule]}, {when} {block_s:g} s",
                        )
            ax.set(
                xscale="log",
                xlabel="E inter-spike interval (ms)",
                ylabel="Fraction of intervals",
                title=f"{recipe.NETWORK_LABELS[network]} · digit {digit}",
            )
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.85)
            legend(ax)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "isi")


def figure_spectra(ctx):
    """E population-rate spectra in the first and the last block."""
    low, high = ctx.cfg["rhythm_band_hz"]
    block_s = ctx.cfg["block_presentations"] * ctx.cfg["presentation_ms"] / 1000
    fig, axes, networks, digits = network_digit_axes(ctx)
    for row, network in enumerate(networks):
        for col, digit in enumerate(digits):
            ax = axes[row, col]
            ax.axvspan(low, high, color=theme.MUTED, alpha=0.08, lw=0)
            for rule in recipe.DESIGN_RULES:
                c = condition(network, digit, rule)
                freqs = ctx.series[f"{c['id']}__E_spectrum_frequencies_hz"]
                power = ctx.series[f"{c['id']}__E_spectrum_power"]
                for index, alpha, when in ((0, 0.4, "first"), (-1, 1.0, "last")):
                    ax.plot(
                        freqs[1:],
                        power[index][1:],
                        ls=dash(c),
                        color=color(ctx, c),
                        alpha=alpha,
                        label=f"{recipe.RULE_LABELS[rule]}, {when} {block_s:g} s",
                    )
            ax.set(
                yscale="log",
                xlim=(0, 1.25 * high),
                xlabel="Frequency (Hz)",
                ylabel="Power of E rate",
                title=f"{recipe.NETWORK_LABELS[network]} · digit {digit}",
            )
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.85)
            legend(ax)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "spectra")


def figure_tendency(ctx):
    """Direction of weight change: signed change, potentiated and depressed shares."""
    seconds = ctx.cfg["presentation_ms"] / 1000
    groups = [
        (title, [c for c in conditions if c["rule"] != "frozen"])
        for title, conditions in panels(ctx)
    ]
    fig, axes = plt.subplots(
        len(groups), 4, figsize=(5.6 * 4, 4.4 * len(groups)), squeeze=False
    )
    for row, (title, conditions) in enumerate(groups):
        signed, digit_share, other_share, by_weight = axes[row]
        for c in conditions:
            w, ink = ctx.results[c["id"]]["weights"], color(ctx, c)
            mean0 = w["mean_us"][0]
            t = np.arange(len(w["mean_us"])) * seconds
            label = f"digit {c['digit']}"
            signed.plot(
                t, np.array(w["digit_net_change"]) / mean0, color=ink, label=label
            )
            signed.plot(t, np.array(w["other_net_change"]) / mean0, color=ink, ls="--")
            digit_share.plot(t, w["digit_potentiated"], color=ink, label=label)
            digit_share.plot(t, w["digit_depressed"], color=ink, ls="--")
            other_share.plot(t, w["other_potentiated"], color=ink, label=label)
            other_share.plot(t, w["other_depressed"], color=ink, ls="--")
            centres, digit_curves = binned(ctx, c["id"], "digit")
            _, other_curves = binned(ctx, c["id"], "other")
            by_weight.plot(
                centres, digit_curves[-1], color=ink, marker="o", ms=3, label=label
            )
            by_weight.plot(centres, other_curves[-1], color=ink, ls="--")
        for ax in (signed, by_weight):
            ax.axhline(0, color=theme.MUTED, lw=0.8)
        by_weight.axvline(
            fixed_point(ctx, conditions[0]["id"]), color=theme.MUTED, ls=":"
        )
        tolerance = 100 * ctx.cfg["change_tolerance_fraction"]
        signed.set(
            xlabel="Time (s)",
            ylabel="Mean ΔW / trained mean",
            title=f"{title}: signed change",
        )
        digit_share.set(
            xlabel="Time (s)",
            ylabel=f"Share changed by > {tolerance:g}%",
            title=f"{title}: digit-pixel synapses",
            ylim=(0, 1),
        )
        other_share.set(
            xlabel="Time (s)",
            ylabel=f"Share changed by > {tolerance:g}%",
            title=f"{title}: other synapses",
            ylim=(0, 1),
        )
        by_weight.set(
            xlabel="Initial weight W0 / w_max",
            ylabel="Mean ΔW / w_max at the end",
            title=f"{title}: change against W0",
        )
        for ax in axes[row]:
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.85)
        style_legend(signed, "digit pixels", "other pixels")
        style_legend(digit_share, "potentiated", "depressed")
        style_legend(other_share, "potentiated", "depressed")
        style_legend(by_weight, "digit pixels", "other pixels")
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "tendency")


def figure_drift(ctx):
    """ΔW against W0 for digit-pixel synapses at every saved time."""
    conditions = [c for c in ctx.cfg["conditions"] if c["rule"] != "frozen"]
    seconds = ctx.cfg["presentation_ms"] / 1000
    fig, axes = grid(len(conditions), columns=len(ctx.cfg["digits"]), width=5.2)
    for ax, c in zip(axes, conditions, strict=True):
        centres, curves = binned(ctx, c["id"], "digit")
        _, others = binned(ctx, c["id"], "other")
        times = ctx.series[f"{c['id']}__snapshot_boundaries"] * seconds
        for k, (curve, when) in enumerate(zip(curves, times, strict=True)):
            x = 0.25 + 0.75 * k / max(len(curves) - 1, 1)
            ax.plot(centres, curve, color=RAMPS["both"](x), label=f"{when:g} s")
        ax.plot(
            centres,
            others[-1],
            color=theme.MUTED,
            ls="--",
            label=f"other pixels, {times[-1]:g} s",
        )
        ax.axhline(0, color=theme.MUTED, lw=0.8)
        ax.axvline(fixed_point(ctx, c["id"]), color=theme.MUTED, ls=":")
        ax.set(
            xlabel="Initial weight W0 / w_max",
            ylabel="Mean ΔW / w_max",
            title=f"{name(ctx, c)}: digit pixels",
        )
        ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.75)
        legend(ax)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "drift")


def figure_maps(ctx):
    """Final against initial weight (W / w_max) and the W / w_max distributions."""
    networks, digits = list(ctx.cfg["source_cells"]), ctx.cfg["digits"]
    rows = [(n, g) for n in networks for g in ("digit", "other")]
    columns = len(digits) + 1
    fig, axes = plt.subplots(
        len(rows), columns, figsize=(4.6 * columns, 4.4 * len(rows)), squeeze=False
    )
    cmap = LinearSegmentedColormap.from_list(
        "density", ["#ffffff", "#8fd9ec", theme.ELECTRIC_CYAN, "#00475a"]
    )
    for r, (network, group) in enumerate(rows):
        label = f"{recipe.NETWORK_LABELS[network]} · {group} pixels"
        spread = axes[r, -1]
        for col, digit in enumerate(digits):
            key = cid(network, digit, "both")
            joint = ctx.series[f"{key}__joint_{group}"]
            edges = ctx.series[f"{key}__edges_{group}"]
            top = float(edges[-1])
            ax = axes[r, col]
            ax.imshow(
                np.log10(joint.T + 1),
                origin="lower",
                extent=[0, top, 0, top],
                cmap=cmap,
                aspect="equal",
                interpolation="nearest",
            )
            ax.plot([0, top], [0, top], color=theme.INK_BLACK, lw=0.8, ls=":")
            fixed = fixed_point(ctx, key)
            if fixed < top:
                ax.axvline(fixed, color=theme.MUTED, lw=0.8, ls="--")
                ax.axhline(fixed, color=theme.MUTED, lw=0.8, ls="--")
            ax.set(
                xlabel="Initial W0 / w_max",
                ylabel="Final W / w_max",
                title=f"{label} · digit {digit}",
            )
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.7)
            ratio = ctx.series[f"{key}__ratio_{group}"]
            centres = (edges[:-1] + edges[1:]) / 2
            if col == 0:
                spread.plot(
                    centres,
                    ratio[0] / ratio[0].sum(),
                    color=theme.MUTED,
                    label="trained",
                )
            spread.plot(
                centres,
                ratio[1] / ratio[1].sum(),
                color=color(ctx, condition(network, digit, "both")),
                label=f"digit {digit}, end",
            )
        spread.set(
            yscale="log",
            xlabel="W / w_max",
            ylabel="Fraction of synapses",
            title=f"{label}: distribution",
        )
        spread.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.7)
        legend(spread)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "maps")


FIGURES = (
    figure_stimulus,
    figure_rasters,
    figure_rhythm,
    figure_autocorrelograms,
    figure_isi,
    figure_spectra,
    figure_activity,
    figure_weights,
    figure_tendency,
    figure_drift,
    figure_maps,
    figure_synapses,
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
    stimulus = dict(np.load(compute.file("dataset.npz"), allow_pickle=False))
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
            "colours": "frozen greys (dashed), STDP + scaling blues; shade = digit",
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
            stimulus=stimulus,
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
