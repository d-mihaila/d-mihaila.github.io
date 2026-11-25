"""Render exp007 figures: rhythm, ISIs, spectra and weight change for frozen, STDP and STDP + scaling."""

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
from experiments.exp007 import recipe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
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

# Hue = rule (frozen black control, STDP only red, STDP + scaling blue);
# shade = digit (light to dark in the order of cfg["digits"]).
RAMPS = {
    "frozen": LinearSegmentedColormap.from_list("frozen", ["#a6a6a6", theme.INK_BLACK]),
    "stdp": LinearSegmentedColormap.from_list(
        "stdp", ["#f2a39b", theme.DEEP_RED, "#5a0f0a"]
    ),
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


PLASTIC = ("stdp", "both")
RULE_SHORT = {"frozen": "frozen", "stdp": "STDP", "both": "STDP + sc."}


def plastic_rows(ctx):
    """(network, rule) rows of grids whose columns are the digits."""
    return [(n, r) for n in ctx.cfg["source_cells"] for r in PLASTIC]


def row_label(network, rule):
    return f"{recipe.NETWORK_LABELS[network]} · {recipe.RULE_LABELS[rule]}"


def figure_rasters(ctx):
    """Per network and digit: the first window, then every rule's last window."""
    pairs = [(n, d) for n in ctx.cfg["source_cells"] for d in ctx.cfg["digits"]]
    rules = recipe.DESIGN_RULES
    fig, axes = plt.subplots(
        len(pairs),
        len(rules) + 1,
        figsize=(4.6 * (len(rules) + 1), 3.6 * len(pairs)),
        squeeze=False,
    )
    for row, (network, digit) in enumerate(pairs):
        arch = ctx.cfg["networks"][network]["architecture"]
        label = f"{recipe.NETWORK_LABELS[network]} · digit {digit}"
        frozen = cid(network, digit, "frozen")
        last = ctx.results[frozen]["presentations"] - 1
        seconds = (last + 1) * ctx.cfg["presentation_ms"] / 1000
        raster(
            axes[row, 0], ctx.spikes[frozen], ctx, arch, 0, f"{label} · first window"
        )
        for col, rule in enumerate(rules, start=1):
            raster(
                axes[row, col],
                ctx.spikes[cid(network, digit, rule)],
                ctx,
                arch,
                last,
                f"{label} · {recipe.RULE_LABELS[rule]}, {seconds:g} s",
            )
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "rasters")


def overlay_figure(ctx, filename, draw, setup):
    """Per network x digit: the frozen first block (dotted grey) and every
    rule's last block, so each rule is read against the same starting point."""
    block_s = ctx.cfg["block_presentations"] * ctx.cfg["presentation_ms"] / 1000
    fig, axes, networks, digits = network_digit_axes(ctx)
    for row, network in enumerate(networks):
        for col, digit in enumerate(digits):
            ax = axes[row, col]
            draw(
                ax,
                condition(network, digit, "frozen"),
                0,
                dict(color=theme.MUTED, ls=":", lw=1.2, label=f"first {block_s:g} s"),
            )
            for rule in recipe.DESIGN_RULES:
                c = condition(network, digit, rule)
                draw(
                    ax,
                    c,
                    -1,
                    dict(
                        color=color(ctx, c),
                        ls=dash(c),
                        label=f"{RULE_SHORT[rule]}, last {block_s:g} s",
                    ),
                )
            setup(ax)
            ax.set_title(f"{recipe.NETWORK_LABELS[network]} · digit {digit}")
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.85)
            legend(ax)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, filename)


def figure_autocorrelograms(ctx):
    """E spike autocorrelograms: frozen first block against every rule's last."""

    def draw(ax, c, index, style):
        lags = ctx.series[f"{c['id']}__E_autocorrelogram_lags_ms"]
        curves = ctx.series[f"{c['id']}__E_autocorrelogram"]
        ax.plot(lags[1:], curves[index][1:], **style)

    def setup(ax):
        ax.axhline(1, color=theme.MUTED, lw=0.6)
        ax.set(xlim=(0, 50), xlabel="Lag (ms)", ylabel="Autocorrelation")

    overlay_figure(ctx, "autocorrelograms", draw, setup)


def figure_spectra(ctx):
    """E population-rate spectra: frozen first block against every rule's last."""
    low, high = ctx.cfg["rhythm_band_hz"]

    def draw(ax, c, index, style):
        freqs = ctx.series[f"{c['id']}__E_spectrum_frequencies_hz"]
        power = ctx.series[f"{c['id']}__E_spectrum_power"]
        ax.plot(freqs[1:], power[index][1:], **style)

    def setup(ax):
        ax.axvspan(low, high, color=theme.MUTED, alpha=0.08, lw=0)
        ax.set(
            yscale="log",
            xlim=(0, 1.25 * high),
            xlabel="Frequency (Hz)",
            ylabel="Power of E rate",
        )

    overlay_figure(ctx, "spectra", draw, setup)


def isi_distribution(ctx, c, index):
    counts = np.array(
        ctx.results[c["id"]]["blocks"]["E"][index]["isi_histogram"], float
    )
    return counts / counts.sum() if counts.sum() else np.full(len(counts), np.nan)


def isi_centres(ctx):
    edges = np.array(ctx.numbers["isi_edges_ms"])
    return np.sqrt(edges[:-1] * edges[1:])


def figure_isi(ctx):
    """E ISI distributions: frozen first block against every rule's last."""
    centres = isi_centres(ctx)

    def draw(ax, c, index, style):
        ax.plot(centres, isi_distribution(ctx, c, index), **style)

    def setup(ax):
        ax.set(
            xscale="log",
            xlabel="E inter-spike interval (ms)",
            ylabel="Fraction of intervals",
        )

    overlay_figure(ctx, "isi", draw, setup)


def figure_isi_relative(ctx):
    """Last-block ISI distribution of each plastic rule minus the frozen one."""
    centres = isi_centres(ctx)
    fig, axes, networks, digits = network_digit_axes(ctx)
    for row, network in enumerate(networks):
        for col, digit in enumerate(digits):
            ax = axes[row, col]
            base = isi_distribution(ctx, condition(network, digit, "frozen"), -1)
            for rule in PLASTIC:
                c = condition(network, digit, rule)
                ax.plot(
                    centres,
                    isi_distribution(ctx, c, -1) - base,
                    color=color(ctx, c),
                    marker="o",
                    ms=3,
                    label=f"{RULE_SHORT[rule]} − frozen",
                )
            ax.axhline(0, color=theme.MUTED, lw=0.8)
            ax.set(
                xscale="log",
                xlabel="E inter-spike interval (ms)",
                ylabel="Δ fraction of intervals",
                title=f"{recipe.NETWORK_LABELS[network]} · digit {digit}",
            )
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.85)
            legend(ax)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "isi_relative")


def figure_rates(ctx):
    """E rate in every 200-ms window against the scaling setpoint."""
    seconds = ctx.cfg["presentation_ms"] / 1000
    fig, axes, networks, digits = network_digit_axes(ctx)
    for row, network in enumerate(networks):
        for col, digit in enumerate(digits):
            ax = axes[row, col]
            for rule in recipe.DESIGN_RULES:
                c = condition(network, digit, rule)
                rates = np.array(ctx.results[c["id"]]["rates_hz"]["E"], float)
                ax.plot(
                    (np.arange(len(rates)) + 0.5) * seconds,
                    rates,
                    color=color(ctx, c),
                    ls=dash(c),
                    label=recipe.RULE_LABELS[rule],
                )
            setpoint = ctx.numbers["rules"][cid(network, digit, "both")][
                "target_rate_hz"
            ]
            if setpoint:
                ax.axhline(
                    setpoint, color=theme.MUTED, ls=":", lw=1.2, label="setpoint ν*"
                )
            ax.set(
                xlabel="Time (s)",
                ylabel="E rate per 200-ms window (Hz)",
                title=f"{recipe.NETWORK_LABELS[network]} · digit {digit}",
            )
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.85)
            legend(ax)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "rates")


def figure_tendency(ctx):
    """Direction of weight change per rule: signed change, shares, and ΔW against W0."""
    seconds = ctx.cfg["presentation_ms"] / 1000
    rows = plastic_rows(ctx)
    fig, axes = plt.subplots(
        len(rows), 4, figsize=(5.6 * 4, 4.2 * len(rows)), squeeze=False
    )
    tolerance = 100 * ctx.cfg["change_tolerance_fraction"]
    for row, (network, rule) in enumerate(rows):
        signed, digit_share, other_share, by_weight = axes[row]
        for digit in ctx.cfg["digits"]:
            c = condition(network, digit, rule)
            w, ink = ctx.results[c["id"]]["weights"], color(ctx, c)
            mean0 = w["mean_us"][0]
            t = np.arange(len(w["mean_us"])) * seconds
            label = f"digit {digit}"
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
            fixed_point(ctx, cid(network, ctx.cfg["digits"][0], rule)),
            color=theme.MUTED,
            ls=":",
        )
        title = row_label(network, rule)
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
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.8)
        style_legend(signed, "digit pixels", "other pixels")
        style_legend(digit_share, "potentiated", "depressed")
        style_legend(other_share, "potentiated", "depressed")
        style_legend(by_weight, "digit pixels", "other pixels")
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "tendency")


def rule_digit_axes(ctx, width=5.2, height=4.2):
    rows, digits = plastic_rows(ctx), ctx.cfg["digits"]
    fig, axes = plt.subplots(
        len(rows),
        len(digits),
        figsize=(width * len(digits), height * len(rows)),
        squeeze=False,
    )
    return fig, axes, rows, digits


def figure_drift(ctx):
    """ΔW against W0 for digit-pixel synapses at every saved time, per rule."""
    seconds = ctx.cfg["presentation_ms"] / 1000
    fig, axes, rows, digits = rule_digit_axes(ctx)
    for r, (network, rule) in enumerate(rows):
        for col, digit in enumerate(digits):
            ax, key = axes[r, col], cid(network, digit, rule)
            centres, curves = binned(ctx, key, "digit")
            _, others = binned(ctx, key, "other")
            times = ctx.series[f"{key}__snapshot_boundaries"] * seconds
            for k, (curve, when) in enumerate(zip(curves, times, strict=True)):
                x = 0.25 + 0.75 * k / max(len(curves) - 1, 1)
                ax.plot(centres, curve, color=RAMPS[rule](x), label=f"{when:g} s")
            ax.plot(
                centres,
                others[-1],
                color=theme.MUTED,
                ls="--",
                label=f"other pixels, {times[-1]:g} s",
            )
            ax.axhline(0, color=theme.MUTED, lw=0.8)
            ax.axvline(fixed_point(ctx, key), color=theme.MUTED, ls=":")
            ax.set(
                xlabel="Initial weight W0 / w_max",
                ylabel="Mean ΔW / w_max",
                title=f"{row_label(network, rule)} · digit {digit}",
            )
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.75)
            legend(ax)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "drift")


def figure_synapses(ctx):
    """Tracked random synapses per rule: digit-pixel synapses in colour, others grey."""
    seconds = ctx.cfg["presentation_ms"] / 1000
    fig, axes, rows, digits = rule_digit_axes(ctx, width=5.0)
    for r, (network, rule) in enumerate(rows):
        for col, digit in enumerate(digits):
            ax, c = axes[r, col], condition(network, digit, rule)
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
                title=f"{row_label(network, rule)} · digit {digit}\n"
                f"{int(on.sum())} of {len(on)} on digit pixels",
            )
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.7)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "synapses")


DENSITY = {
    rule: LinearSegmentedColormap.from_list(
        f"density-{rule}", ["#ffffff", RAMPS[rule](0.35), RAMPS[rule](1.0)]
    )
    for rule in ("stdp", "both")
}


def maps_figure(ctx, group):
    """Final against initial weight (W / w_max) per rule and digit, for one pixel group."""
    fig, axes, rows, digits = rule_digit_axes(ctx, width=4.6, height=4.4)
    for r, (network, rule) in enumerate(rows):
        for col, digit in enumerate(digits):
            ax, key = axes[r, col], cid(network, digit, rule)
            joint = ctx.series[f"{key}__joint_{group}"]
            top = float(ctx.series[f"{key}__edges_{group}"][-1])
            ax.imshow(
                np.log10(joint.T + 1),
                origin="lower",
                extent=[0, top, 0, top],
                cmap=DENSITY[rule],
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
                title=f"{row_label(network, rule)} · digit {digit}",
            )
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.7)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, f"maps_{group}")


def figure_maps_digit(ctx):
    maps_figure(ctx, "digit")


def figure_maps_other(ctx):
    maps_figure(ctx, "other")


def figure_distributions(ctx):
    """W / w_max distributions, trained and at the end, per network, group and rule."""
    networks, digits = list(ctx.cfg["source_cells"]), ctx.cfg["digits"]
    columns = [(g, r) for g in ("digit", "other") for r in PLASTIC]
    fig, axes = plt.subplots(
        len(networks),
        len(columns),
        figsize=(5.2 * len(columns), 4.2 * len(networks)),
        squeeze=False,
    )
    for row, network in enumerate(networks):
        for col, (group, rule) in enumerate(columns):
            ax = axes[row, col]
            for k, digit in enumerate(digits):
                key = cid(network, digit, rule)
                ratio = ctx.series[f"{key}__ratio_{group}"]
                edges = ctx.series[f"{key}__edges_{group}"]
                centres = (edges[:-1] + edges[1:]) / 2
                if k == 0:
                    ax.plot(
                        centres,
                        ratio[0] / ratio[0].sum(),
                        color=theme.MUTED,
                        label="trained",
                    )
                ax.plot(
                    centres,
                    ratio[1] / ratio[1].sum(),
                    color=color(ctx, condition(network, digit, rule)),
                    label=f"digit {digit}, end",
                )
            ax.set(
                yscale="log",
                xlabel="W / w_max",
                ylabel="Fraction of synapses",
                title=f"{row_label(network, rule)} · {group} pixels",
            )
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.75)
            legend(ax)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "distributions")


def figure_pairs(ctx):
    """Per synapse: ΔW with STDP + scaling against ΔW with STDP only (same input)."""
    networks, digits = list(ctx.cfg["source_cells"]), ctx.cfg["digits"]
    rows = [(n, g) for n in networks for g in ("digit", "other")]
    fig, axes = plt.subplots(
        len(rows),
        len(digits),
        figsize=(4.6 * len(digits), 4.4 * len(rows)),
        squeeze=False,
    )
    for r, (network, group) in enumerate(rows):
        for col, digit in enumerate(digits):
            ax = axes[r, col]
            key = f"{network}__{recipe.stream_kind(digit)}"
            joint = ctx.series[f"{key}__scaling_joint_{group}"]
            edges = ctx.series[f"{key}__scaling_edges_{group}"]
            low, high = float(edges[0]), float(edges[-1])
            ax.imshow(
                np.log10(joint.T + 1),
                origin="lower",
                extent=[low, high, low, high],
                cmap=DENSITY["both"],
                aspect="equal",
                interpolation="nearest",
            )
            ax.plot([low, high], [low, high], color=theme.INK_BLACK, lw=0.8, ls=":")
            ax.axhline(0, color=theme.MUTED, lw=0.6)
            ax.axvline(0, color=theme.MUTED, lw=0.6)
            effect = ctx.numbers["scaling_effect"][key]
            ax.set(
                xlabel="ΔW / w_max, STDP only",
                ylabel="ΔW / w_max, STDP + scaling",
                title=f"{recipe.NETWORK_LABELS[network]} · digit {digit} · {group} pixels\n"
                f"scaling adds {100 * effect[f'{group}_extra_change']:+.1f}% of the trained mean",
            )
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.7)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "pairs")


def figure_cells(ctx):
    """Per E cell: what scaling did to its total input, against its own rate."""
    networks, digits = list(ctx.cfg["source_cells"]), ctx.cfg["digits"]
    fig, axes = plt.subplots(
        len(networks), 3, figsize=(5.8 * 3, 4.6 * len(networks)), squeeze=False
    )
    rate_edges = np.logspace(-2, 1.5, 36)
    for row, network in enumerate(networks):
        ratio_ax, rates_ax, total_ax = axes[row]
        for digit in digits:
            key = f"{network}__{recipe.stream_kind(digit)}"
            ink = shade(ctx, "both", digit)
            x = ctx.series[f"{key}__stdp_rate_over_setpoint"]
            y = ctx.series[f"{key}__afferent_ratio"]
            alive = x > 0
            ratio_ax.scatter(
                x[alive],
                y[alive],
                s=6,
                color=ink,
                alpha=0.5,
                linewidths=0,
                label=f"digit {digit}",
            )
            setpoint = ctx.numbers["scaling_effect"][key]["setpoint_hz"]
            for rule in recipe.DESIGN_RULES:
                c = condition(network, digit, rule)
                rates = ctx.series[f"{c['id']}__E_cell_rate_last"] / setpoint
                counts, _ = np.histogram(
                    np.clip(rates, rate_edges[0], rate_edges[-1]), bins=rate_edges
                )
                rates_ax.plot(
                    np.sqrt(rate_edges[:-1] * rate_edges[1:]),
                    counts / counts.sum(),
                    color=color(ctx, c),
                    ls=dash(c),
                    label=RULE_SHORT[rule] if digit == digits[0] else None,
                )
            for rule in PLASTIC:
                c = condition(network, digit, rule)
                total = (
                    ctx.series[f"{c['id']}__afferent_last"]
                    / ctx.series[f"{c['id']}__afferent_first"]
                )
                counts, edges = np.histogram(total, bins=40, range=(0, 3))
                total_ax.plot(
                    (edges[:-1] + edges[1:]) / 2,
                    counts / counts.sum(),
                    color=color(ctx, c),
                    label=f"digit {digit} · {RULE_SHORT[rule]}",
                )
        ratio_ax.axhline(1, color=theme.MUTED, lw=0.8)
        ratio_ax.axvline(1, color=theme.MUTED, lw=0.8, ls=":")
        ratio_ax.set(
            xscale="log",
            xlabel="Cell rate with STDP only / setpoint ν*",
            ylabel="Total input: STDP + scaling / STDP only",
            title=f"{recipe.NETWORK_LABELS[network]}: scaling per E cell",
        )
        rates_ax.axvline(1, color=theme.MUTED, lw=0.8, ls=":")
        rates_ax.set(
            xscale="log",
            xlabel="E cell rate in the last second / ν*",
            ylabel="Fraction of E cells",
            title=f"{recipe.NETWORK_LABELS[network]}: per-cell rates",
        )
        total_ax.axvline(1, color=theme.MUTED, lw=0.8, ls=":")
        total_ax.set(
            xlabel="Total input weight at the end / trained",
            ylabel="Fraction of E cells",
            title=f"{recipe.NETWORK_LABELS[network]}: per-cell input change",
        )
        for ax in (ratio_ax, rates_ax):
            ax.xaxis.set_minor_formatter(NullFormatter())
        for ax in axes[row]:
            ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.85)
        ratio_ax.legend(frameon=False, fontsize=SMALL, markerscale=3)
        legend(rates_ax)
        legend(total_ax)
    tidy(axes)
    fig.tight_layout()
    save(fig, ctx.export, "cells")


SUMMARY_CHANGES = (
    ("rhythm_contrast", "Δ rhythmicity contrast", 1.0),
    ("rhythm_frequency_hz", "Δ rhythm frequency (Hz)", 1.0),
    ("participation", "Δ E cells firing per cycle", 1.0),
    ("rate_hz", "Δ E rate (Hz)", 1.0),
    ("isi_cv_mean", "Δ mean E ISI CV", 1.0),
    ("spectral_peak_hz", "Δ spectral peak (Hz)", 1.0),
)
SUMMARY_WEIGHTS = (
    ("digit_net_change_last", "Digit-pixel ΔW (% trained mean)", 100.0),
    ("other_net_change_last", "Other-pixel ΔW (% trained mean)", 100.0),
    ("weight_change_last", "Mean |ΔW| (% trained mean)", 100.0),
    ("weight_correlation_last", "r with trained W at the end", 1.0),
)


def summary_value(row, key, scale, change):
    if change:
        first, last = row.get(f"{key}_first"), row.get(f"{key}_last")
        return np.nan if first is None or last is None else scale * (last - first)
    value = row.get(key)
    return np.nan if value is None else scale * value


def summary_panel(ctx, ax, key, label, scale, rules, change):
    from matplotlib.transforms import blended_transform_factory

    networks, digits = list(ctx.cfg["source_cells"]), ctx.cfg["digits"]
    spacing = len(rules) + 1
    ticks, labels = [], []
    for n_i, network in enumerate(networks):
        for r_i, rule in enumerate(rules):
            x0 = n_i * spacing + r_i
            found = []
            for k, digit in enumerate(digits):
                row = ctx.numbers["summary"][cid(network, digit, rule)]
                value = summary_value(row, key, scale, change)
                offset = 0.12 * (k - (len(digits) - 1) / 2)
                ax.plot(x0 + offset, value, "o", color=shade(ctx, rule, digit), ms=6)
                found.append(value)
            if np.isfinite(found).any():
                ax.hlines(
                    np.nanmean(found), x0 - 0.3, x0 + 0.3, color=RAMPS[rule](0.7), lw=2
                )
            ticks.append(x0)
            labels.append(RULE_SHORT[rule])
        middle = n_i * spacing + (len(rules) - 1) / 2
        ax.text(
            middle,
            -0.3,
            recipe.NETWORK_LABELS[network],
            transform=blended_transform_factory(ax.transData, ax.transAxes),
            ha="center",
            fontsize=SMALL,
        )
    ax.axhline(0, color=theme.MUTED, lw=0.6)
    ax.set_xticks(ticks, labels, rotation=30, fontsize=SMALL)
    ax.set(ylabel=label, title=label)
    ax.title.set_fontsize(theme.SIZE_TITLE * FONT_SCALE * 0.75)


def figure_summary(ctx):
    """Every measure per rule and network; one dot per digit, bar = mean of digits."""
    panels_needed = len(SUMMARY_CHANGES) + len(SUMMARY_WEIGHTS)
    fig, axes = plt.subplots(2, 5, figsize=(5.2 * 5, 5.0 * 2))
    flat = list(axes.ravel())
    for ax, (key, label, scale) in zip(flat, SUMMARY_CHANGES, strict=False):
        summary_panel(ctx, ax, key, label, scale, recipe.DESIGN_RULES, change=True)
    for ax, (key, label, scale) in zip(
        flat[len(SUMMARY_CHANGES) :], SUMMARY_WEIGHTS, strict=False
    ):
        summary_panel(ctx, ax, key, label, scale, PLASTIC, change=False)
    for ax in flat[panels_needed:]:
        ax.set_visible(False)
    tidy(flat[:panels_needed])
    fig.tight_layout()
    save(fig, ctx.export, "summary")


FIGURES = (
    figure_stimulus,
    figure_rasters,
    figure_rates,
    figure_rhythm,
    figure_autocorrelograms,
    figure_spectra,
    figure_isi,
    figure_isi_relative,
    figure_activity,
    figure_weights,
    figure_tendency,
    figure_drift,
    figure_pairs,
    figure_cells,
    figure_maps_digit,
    figure_maps_other,
    figure_distributions,
    figure_synapses,
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
            "colours": "frozen greys (dashed), STDP only reds, STDP + scaling blues; shade = digit",
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
