"""Render exp176 diagrams, rasters, weight, rate, ISI, spectrum, receptive-field and accuracy figures."""

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
from experiments.exp176 import recipe
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.font_manager import FontProperties
from pingstore.library import source_run, stage_run
from snnlab import lang, viz

FONT_SCALE = 1.5
# Colour = plasticity condition (frozen: black control; van Rossum: theme blue).
# Line style and marker = network (PING solid circles, COBA dashed squares).
CONDITION_COLOR = {"frozen": theme.INK_BLACK, "van_rossum": theme.ELECTRIC_CYAN}
NETWORK_STYLE = {"ping": dict(ls="-", marker="o"), "coba": dict(ls="--", marker="s")}


def line(network, condition="van_rossum", marker=False):
    """Line style for one network and condition."""
    style = dict(color=CONDITION_COLOR[condition], ls=NETWORK_STYLE[network]["ls"])
    if marker:
        style["marker"] = NETWORK_STYLE[network]["marker"]
    return style


def label(network, condition=None):
    text = recipe.LABELS[network]
    return text if condition is None else f"{text} · {recipe.LABELS[condition]}"


def circuit(bundle, path, network):
    """House-style native circuit; the plastic sensory projection is marked."""
    native = lang.diagram(bundle, view="circuit")
    symbols = {"drive": "X", "E": "E", "I": "I", "readout": "R"}
    diagram = replace(
        native,
        title=f"{recipe.LABELS[network]} (exp022 TR01) with van Rossum plasticity on X→E",
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
    for ax in np.atleast_1d(axes).ravel():
        ax.spines[["top", "right"]].set_visible(False)
    for letter in theme.label_panels(list(np.atleast_1d(axes).ravel())):
        letter.set_fontsize(theme.SIZE_PANEL * FONT_SCALE)


def raster(ax, spikes, cfg, network, presentation, title):
    """I (red) above E (black), dots; one 200-ms presentation."""
    start = presentation * cfg["presentation_ms"]
    stop = start + cfg["presentation_ms"]
    offset = {"E": 0, "I": cfg["n_e"]}
    for p, color in (("E", theme.INK_BLACK), ("I", theme.DEEP_RED)):
        if p not in recipe.populations(network):
            continue
        t, c = spikes[p + "_time_ms"], spikes[p + "_cell"]
        keep = (t >= start) & (t < stop)
        ax.scatter(
            t[keep] - start,
            c[keep] + offset[p],
            s=0.8,
            c=color,
            linewidths=0,
            rasterized=True,
        )
    top = cfg["n_e"] + (cfg["n_i"] if network == "ping" else 0)
    if network == "ping":
        ax.axhline(cfg["n_e"] - 0.5, color=theme.MUTED, lw=0.5)
    ax.set(
        xlim=(0, cfg["presentation_ms"]),
        ylim=(-0.5, top - 0.5),
        title=title,
        xlabel="Time in presentation (ms)",
        ylabel="Neuron (E below I)" if network == "ping" else "E neuron",
    )
    ax.tick_params(axis="x", labelbottom=True)


def figure_rasters(ctx):
    """Eoin's starting point: frozen TR01 rasters, same first presentation."""
    export, numbers, cfg, spikes = ctx.export, ctx.numbers, ctx.cfg, ctx.spikes
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    digit = numbers["stream_labels"][0]
    for ax, network in zip(axes, recipe.NETWORKS, strict=True):
        raster(
            ax,
            spikes[(network, "frozen")],
            cfg,
            network,
            0,
            f"{recipe.LABELS[network]} (TR01, no rate penalty)",
        )
    fig.suptitle(
        f"Trained exp022 networks, first stream image (digit {digit}), fresh state"
    )
    tidy(axes)
    fig.tight_layout()
    save(fig, export, "rasters")


def figure_rasters_evolution(ctx):
    """Rasters at the start and end of the stream, frozen vs van Rossum."""
    export, cfg, spikes, n_pres = ctx.export, ctx.cfg, ctx.spikes, ctx.n_pres
    last = n_pres - 1
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.4), sharex=True)
    for row, network in enumerate(recipe.NETWORKS):
        for col, (condition, presentation, title) in enumerate(
            (
                ("frozen", 0, "Presentation 1"),
                ("frozen", last, f"Presentation {n_pres}, frozen"),
                ("van_rossum", last, f"Presentation {n_pres}, van Rossum"),
            )
        ):
            raster(
                axes[row, col],
                spikes[(network, condition)],
                cfg,
                network,
                presentation,
                f"{recipe.LABELS[network]} · {title}",
            )
    tidy(axes)
    fig.tight_layout()
    save(fig, export, "rasters-evolution")


def figure_weights(ctx):
    """Weight evolution of the plastic sensory matrix."""
    export, numbers, results, series, time_s = (
        ctx.export,
        ctx.numbers,
        ctx.results,
        ctx.series,
        ctx.time_s,
    )
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.4))
    for network in recipe.NETWORKS:
        w = results[network]["van_rossum"]["weights"]
        rule = numbers["rules"][network]
        axes[0, 0].plot(
            time_s,
            np.array(w["mean_us"]) / rule["initial_mean_us"],
            **line(network),
            label=recipe.LABELS[network],
        )
        axes[0, 1].plot(
            time_s,
            np.array(w["mean_abs_change_us"]) / rule["initial_mean_us"],
            **line(network),
            label=recipe.LABELS[network],
        )
        axes[0, 2].plot(
            time_s,
            w["correlation_with_initial"],
            **line(network),
            label=recipe.LABELS[network],
        )
    axes[0, 0].axhline(1, color=theme.MUTED, lw=0.8, ls=":")
    axes[0, 0].set(
        xlabel="Stream time (s)",
        ylabel="Mean weight / initial mean",
        title="Mean X→E weight",
    )
    axes[0, 1].set(
        xlabel="Stream time (s)",
        ylabel="Mean |ΔW| / initial mean",
        title="Accumulated change",
    )
    axes[0, 2].set(
        xlabel="Stream time (s)",
        ylabel="Pearson r with initial W",
        title="Memory of trained weights",
    )
    for ax in axes[0]:
        ax.legend(frameon=False)
    for col, network in enumerate(recipe.NETWORKS):
        hist = series[f"{network}__van_rossum__weight_histogram"].astype(float)
        edges = series[f"{network}__van_rossum__weight_histogram_edges"]
        ax = axes[1, col]
        image = ax.imshow(
            hist.T + 1,
            origin="lower",
            aspect="auto",
            extent=(time_s[0], time_s[-1], edges[0], edges[-1]),
            norm=LogNorm(),
            cmap="Greys",
        )
        ax.set(
            xlabel="Stream time (s)",
            ylabel="Weight (µS)",
            title=f"{recipe.LABELS[network]} weight histogram",
        )
        fig.colorbar(image, ax=ax, label="Synapses + 1")
    ax = axes[1, 2]
    for network in recipe.NETWORKS:
        hist = series[f"{network}__van_rossum__weight_histogram"].astype(float)
        edges = series[f"{network}__van_rossum__weight_histogram_edges"]
        centres = 0.5 * (edges[1:] + edges[:-1]) / edges[-1]
        for index, condition in ((0, "frozen"), (-1, "van_rossum")):
            ax.plot(
                centres,
                np.maximum(hist[index], 0.5),
                label=f"{recipe.LABELS[network]} {'start' if index == 0 else 'end'}",
                **line(network, condition),
            )
    ax.set_yscale("log")
    ax.set(
        xlabel="Weight / w_max",
        ylabel="Synapses",
        title="Start (black) vs end (blue)",
    )
    ax.legend(frameon=False)
    tidy(axes)
    fig.tight_layout()
    save(fig, export, "weights")


def figure_activity(ctx):
    """Firing rates and the scaling sensor."""
    export, numbers, results, time_s, mid_s = (
        ctx.export,
        ctx.numbers,
        ctx.results,
        ctx.time_s,
        ctx.mid_s,
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for network in recipe.NETWORKS:
        for condition in recipe.CONDITIONS:
            rates = results[network][condition]["rates_hz"]
            style = line(network, condition)
            axes[0].plot(mid_s, rates["E"], label=label(network, condition), **style)
            if "I" in rates:
                axes[1].plot(
                    mid_s, rates["I"], label=label(network, condition), **style
                )
        w = results[network]["van_rossum"]["weights"]
        axes[2].plot(
            time_s[1:],
            w["scaling_rate_mean_hz"],
            **line(network),
            label=f"{recipe.LABELS[network]} sensor",
        )
        axes[2].axhline(
            numbers["scaling_target_rate_hz"][network],
            color=theme.INK_BLACK,
            ls=":",
            lw=1,
            label=f"{recipe.LABELS[network]} target",
        )
    axes[0].set(
        xlabel="Stream time (s)", ylabel="E rate (Hz)", title="Excitatory firing"
    )
    axes[1].set(
        xlabel="Stream time (s)",
        ylabel="I rate (Hz)",
        title="Inhibitory firing (PING only)",
    )
    axes[2].set(
        xlabel="Stream time (s)",
        ylabel="Rate estimate (Hz)",
        title="Synaptic-scaling sensor",
    )
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=theme.SIZE_LEGEND * FONT_SCALE * 0.85)
    tidy(axes)
    fig.tight_layout()
    save(fig, export, "activity")


def isi_density(row, edges):
    counts = np.array(row["isi_histogram"], dtype=float)
    return counts / max(counts.sum(), 1) / np.diff(np.log10(edges))


def figure_isi(ctx):
    """Excitatory inter-spike intervals, block by block."""
    export, numbers, results, block_s, block_mid_s = (
        ctx.export,
        ctx.numbers,
        ctx.results,
        ctx.block_s,
        ctx.block_mid_s,
    )
    edges = np.array(numbers["isi_edges_ms"])
    centres = np.sqrt(edges[1:] * edges[:-1])
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.2), layout="constrained")
    # Top row: ISI density in every block, one column per network and condition.
    pairs = [(n, c) for n in recipe.NETWORKS for c in recipe.CONDITIONS]
    peak = max(
        isi_density(row, edges).max()
        for n, c in pairs
        for row in results[n][c]["blocks"]["E"]
    )
    for ax, (network, condition) in zip(axes[0], pairs, strict=True):
        rows = results[network][condition]["blocks"]["E"]
        grid = np.stack([isi_density(row, edges) for row in rows]).T
        image = ax.pcolormesh(
            np.arange(len(rows) + 1) * block_s,
            edges,
            grid,
            cmap="Greys",
            vmin=0,
            vmax=peak,
            shading="flat",
            rasterized=True,
        )
        ax.set_yscale("log")
        ax.set(
            xlabel="Stream time (s)",
            ylabel="E ISI (ms)",
            title=f"{label(network, condition)}",
        )
    fig.colorbar(image, ax=axes[0].tolist(), label="Density per log10 ms", pad=0.01)
    # Bottom row: first versus last block, then CV and median per block.
    for col, network in enumerate(recipe.NETWORKS):
        ax = axes[1, col]
        for condition in recipe.CONDITIONS:
            rows = results[network][condition]["blocks"]["E"]
            for index, alpha in ((0, 0.45), (-1, 1.0)):
                ax.plot(
                    centres,
                    isi_density(rows[index], edges),
                    color=CONDITION_COLOR[condition],
                    alpha=alpha,
                    label=f"{recipe.LABELS[condition]}, "
                    f"{'first' if index == 0 else 'last'} {block_s:g} s",
                )
        ax.set_xscale("log")
        ax.set(
            xlabel="E inter-spike interval (ms)",
            ylabel="Density per log10 ms",
            title=f"{recipe.LABELS[network]} first vs last block",
        )
        ax.legend(frameon=False, fontsize=theme.SIZE_LEGEND * FONT_SCALE * 0.75)
    for network in recipe.NETWORKS:
        for condition in recipe.CONDITIONS:
            rows = results[network][condition]["blocks"]["E"]
            style = line(network, condition, marker=True)
            cv = [
                np.nan if r["isi_cv_mean"] is None else r["isi_cv_mean"] for r in rows
            ]
            median = [
                np.nan if r["isi_median_ms"] is None else r["isi_median_ms"]
                for r in rows
            ]
            axes[1, 2].plot(block_mid_s, cv, label=label(network, condition), **style)
            axes[1, 3].plot(
                block_mid_s, median, label=label(network, condition), **style
            )
    axes[1, 2].set(
        xlabel="Stream time (s)",
        ylabel="Mean ISI CV",
        title="Irregularity of E spiking",
    )
    axes[1, 3].set(
        xlabel="Stream time (s)", ylabel="Median E ISI (ms)", title="Typical E interval"
    )
    for ax in axes[1, 2:]:
        ax.legend(frameon=False, fontsize=theme.SIZE_LEGEND * FONT_SCALE * 0.75)
    tidy(axes)
    save(fig, export, "isi")


def figure_synapses(ctx):
    """Trajectories of the tracked random synapses."""
    export, numbers, series, time_s = ctx.export, ctx.numbers, ctx.series, ctx.time_s
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))
    for ax, network in zip(axes[:2], recipe.NETWORKS, strict=True):
        tracks = series[f"{network}__van_rossum__tracked_weights"]
        rule = numbers["rules"][network]
        order = np.argsort(tracks[0])
        shades = np.linspace(0.25, 1.0, len(order))
        for rank, synapse in enumerate(order):
            ax.plot(
                time_s,
                tracks[:, synapse],
                color=CONDITION_COLOR["van_rossum"],
                alpha=0.08 + 0.4 * shades[rank] ** 4,
                lw=0.6,
            )
        ax.axhline(rule["equilibrium_us"], color=theme.MUTED, ls=":", lw=1.2)
        ax.set(
            xlabel="Stream time (s)",
            ylabel="Weight (µS)",
            title=f"{recipe.LABELS[network]}: {tracks.shape[1]} random X→E synapses",
        )
    ax = axes[2]
    for network in recipe.NETWORKS:
        tracks = series[f"{network}__van_rossum__tracked_weights"]
        w_max = numbers["rules"][network]["w_max_us"]
        ax.scatter(
            tracks[0] / w_max,
            (tracks[-1] - tracks[0]) / w_max,
            s=8,
            color=CONDITION_COLOR["van_rossum"]
            if network == "ping"
            else theme.INK_BLACK,
            marker=NETWORK_STYLE[network]["marker"],
            alpha=0.6,
            linewidths=0,
            label=recipe.LABELS[network],
        )
        ax.axvline(
            numbers["rules"][network]["equilibrium_us"] / w_max,
            color=CONDITION_COLOR["van_rossum"]
            if network == "ping"
            else theme.INK_BLACK,
            ls=":",
            lw=1.2,
        )
    ax.axhline(0, color=theme.MUTED, lw=0.8)
    ax.set(
        xlabel="Initial weight / w_max",
        ylabel="Change by the end / w_max",
        title="Change versus initial weight",
    )
    ax.legend(frameon=False)
    tidy(axes)
    fig.tight_layout()
    save(fig, export, "synapses")


def series_of(rows, key):
    return [np.nan if r.get(key) is None else r[key] for r in rows]


def figure_spectra(ctx):
    """Population rhythm: autocorrelograms, rhythmicity contrast and spectra."""
    export, cfg, results, series, block_s, block_mid_s = (
        ctx.export,
        ctx.cfg,
        ctx.results,
        ctx.series,
        ctx.block_s,
        ctx.block_mid_s,
    )
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.2))
    band = cfg["rhythm_band_hz"]
    for col, network in enumerate(recipe.NETWORKS):
        for condition in recipe.CONDITIONS:
            lags = series[f"{network}__{condition}__E_autocorrelogram_lags_ms"]
            ac = series[f"{network}__{condition}__E_autocorrelogram"]
            f = series[f"{network}__{condition}__E_spectrum_frequencies_hz"]
            power = series[f"{network}__{condition}__E_spectrum_power"]
            for index, alpha in ((0, 0.4), (-1, 1.0)):
                text = f"{recipe.LABELS[condition]}, {'first' if index == 0 else 'last'} {block_s:g} s"
                axes[0, col].plot(
                    lags[1:],
                    ac[index][1:],
                    color=CONDITION_COLOR[condition],
                    alpha=alpha,
                    label=text,
                )
                axes[1, col].plot(
                    f[1:],
                    power[index][1:],
                    color=CONDITION_COLOR[condition],
                    alpha=alpha,
                    label=text,
                )
        axes[0, col].axhline(1, color=theme.MUTED, lw=0.8, ls=":")
        axes[0, col].set(
            xlim=(0, 50),
            xlabel="Lag (ms)",
            ylabel="Autocorrelation (1 = independent)",
            title=f"{recipe.LABELS[network]} E spike autocorrelogram",
        )
        axes[1, col].axvspan(*band, color=theme.RULE, zorder=0)
        axes[1, col].set_yscale("log")
        axes[1, col].set(
            xlim=(0, 250),
            xlabel="Frequency (Hz)",
            ylabel="PSD of E rate (Hz²/Hz)",
            title=f"{recipe.LABELS[network]} E population spectrum",
        )
        for ax in axes[:, col]:
            ax.legend(frameon=False, fontsize=theme.SIZE_LEGEND * FONT_SCALE * 0.75)
    for network in recipe.NETWORKS:
        for condition in recipe.CONDITIONS:
            rows = results[network][condition]["blocks"]["E"]
            style = line(network, condition, marker=True)
            name = label(network, condition)
            axes[0, 2].plot(
                block_mid_s, series_of(rows, "rhythm_contrast"), label=name, **style
            )
            axes[0, 3].plot(
                block_mid_s, series_of(rows, "rhythm_frequency_hz"), label=name, **style
            )
            axes[1, 2].plot(
                block_mid_s, series_of(rows, "rhythm_peak_hz"), label=name, **style
            )
            axes[1, 3].plot(
                block_mid_s, series_of(rows, "rhythm_fraction"), label=name, **style
            )
    axes[0, 2].set(
        ylim=(0, 1),
        xlabel="Stream time (s)",
        ylabel="Contrast (lobe − trough)/(lobe + trough)",
        title="Rhythmicity of E (exp054 metric)",
    )
    axes[0, 3].set(
        xlabel="Stream time (s)",
        ylabel="1 / side-peak lag (Hz)",
        title="Rhythm frequency (autocorrelogram)",
    )
    axes[1, 2].set(
        xlabel="Stream time (s)",
        ylabel="Peak frequency (Hz)",
        title=f"Spectral peak in {band[0]:g}–{band[1]:g} Hz",
    )
    axes[1, 3].set(
        xlabel="Stream time (s)",
        ylabel="Fraction of power",
        title=f"{band[0]:g}–{band[1]:g} Hz power share",
    )
    for ax in (axes[0, 2], axes[0, 3], axes[1, 2], axes[1, 3]):
        ax.legend(frameon=False, fontsize=theme.SIZE_LEGEND * FONT_SCALE * 0.75)
    tidy(axes)
    fig.tight_layout()
    save(fig, export, "spectra")


def figure_receptive_fields(ctx):
    """Receptive fields of the most strongly driven E neurons."""
    export, cfg, series = ctx.export, ctx.cfg, ctx.series
    count = cfg["receptive_fields"]
    fig, axes = plt.subplots(
        2 * 3, count, figsize=(2.6 * count, 2.7 * 6), squeeze=False
    )
    for block, network in enumerate(recipe.NETWORKS):
        first = series[f"{network}__van_rossum__rf_initial"]
        final = series[f"{network}__van_rossum__rf_final"]
        neurons = series[f"{network}__van_rossum__rf_neurons"]
        top = max(first.max(), final.max())
        for j in range(min(count, len(neurons))):
            change = final[j] - first[j]
            limit = max(abs(change).max(), 1e-12)
            for row, (image, title, kwargs) in enumerate(
                (
                    (first[j], "start", dict(cmap="Greys", vmin=0, vmax=top)),
                    (final[j], "end", dict(cmap="Greys", vmin=0, vmax=top)),
                    (
                        change,
                        "end − start",
                        dict(cmap="RdBu", norm=TwoSlopeNorm(0, -limit, limit)),
                    ),
                )
            ):
                ax = axes[3 * block + row, j]
                ax.imshow(image.reshape(28, 28), **kwargs)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(
                    f"{recipe.LABELS[network]} E{neurons[j]} {title}",
                    fontsize=theme.SIZE_LABEL * FONT_SCALE * 0.75,
                )
    fig.suptitle(
        "X→E receptive fields (28 × 28) of the E cells\nwith the largest trained input weights"
    )
    fig.tight_layout()
    save(fig, export, "receptive-fields")


def figure_accuracy(ctx):
    """Held-out test accuracy during the plastic stream."""
    export, cfg, results = ctx.export, ctx.cfg, ctx.results
    seconds = cfg["presentation_ms"] / 1000
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.6, 1]}
    )
    ax = axes[0]
    for network in recipe.NETWORKS:
        frozen = results[network]["frozen"]["accuracy"]
        plastic = results[network]["van_rossum"]["accuracy"]
        if frozen["percent"] is not None:
            ax.axhline(
                frozen["percent"],
                label=label(network, "frozen"),
                **line(network, "frozen"),
            )
        times = np.array(plastic["boundaries"]) * seconds
        values = [np.nan if v is None else v for v in plastic["percent_curve"]]
        ax.plot(
            times,
            values,
            label=label(network, "van_rossum"),
            **line(network, "van_rossum", marker=True),
        )
    images = results["ping"]["frozen"]["accuracy"]["images"]
    ax.set(
        xlabel="Stream time (s)",
        ylabel="Test accuracy (%)",
        title=f"{images} official test images, every {cfg['accuracy_every']} digits",
    )
    ax.legend(frameon=False)
    ax = axes[1]
    positions, heights, colors, hatches, names = [], [], [], [], []
    for i, network in enumerate(recipe.NETWORKS):
        curve = results[network]["van_rossum"]["accuracy"]["percent_curve"]
        for j, (value, condition, name) in enumerate(
            ((curve[0], "frozen", "trained W"), (curve[-1], "van_rossum", "after vR"))
        ):
            positions.append(3 * i + j)
            heights.append(np.nan if value is None else value)
            colors.append(CONDITION_COLOR[condition])
            hatches.append("" if network == "ping" else "//")
            names.append(f"{recipe.LABELS[network]}\n{name}")
    bars = ax.bar(positions, heights, color=colors, edgecolor=theme.PAPER)
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)
    for x, h in zip(positions, heights, strict=True):
        if np.isfinite(h):
            ax.annotate(
                f"{h:.1f}%",
                (x, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )
    ax.set_xticks(positions, names)
    ax.set(ylim=(0, 105), ylabel="Test accuracy (%)", title="Start versus end")
    tidy(axes)
    fig.tight_layout()
    save(fig, export, "accuracy")


FIGURES = (
    figure_rasters,
    figure_rasters_evolution,
    figure_weights,
    figure_synapses,
    figure_activity,
    figure_isi,
    figure_spectra,
    figure_receptive_fields,
    figure_accuracy,
)


def present(identity):
    started = time.perf_counter()
    source = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="analyse"
    )
    numbers = json.loads(source.file("numbers.json").read_text())
    cfg, results = numbers["configuration"], numbers["results"]
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
        (n, c): dict(np.load(compute.file(n, c, "spikes.npz"), allow_pickle=False))
        for n in recipe.NETWORKS
        for c in recipe.CONDITIONS
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
    n_pres = cfg["presentations"]
    time_s = np.arange(n_pres + 1) * cfg["presentation_ms"] / 1000
    mid_s = (np.arange(n_pres) + 0.5) * cfg["presentation_ms"] / 1000
    block_s = cfg["isi_block_presentations"] * cfg["presentation_ms"] / 1000
    blocks = n_pres // cfg["isi_block_presentations"]
    block_mid_s = (np.arange(blocks) + 0.5) * block_s
    with stage_run(
        ROOT,
        recipe.SLUG,
        "present",
        inputs={"analysis": source},
        configuration={
            "style": "experiments.theme paper; enlarged article typography",
            "font_scale": FONT_SCALE,
            "colours": "PING black, COBA red; frozen dashed, van Rossum solid",
            "figures": [
                "rasters",
                "rasters-evolution",
                "weights",
                "synapses",
                "activity",
                "isi",
                "spectra",
                "receptive-fields",
                "accuracy",
            ],
        },
    ) as run:
        export = run.export
        shutil.copyfile(source.file("numbers.json"), export / "numbers.json")
        for network in recipe.NETWORKS:
            bundle_path = compute.restore_bundle(
                f"{network}/network.bundle", run.scratch / f"{network}.bundle"
            )
            circuit(
                lang.load_bundle(bundle_path),
                export / f"circuit-{network}.svg",
                network,
            )

        ctx = SimpleNamespace(
            export=export,
            numbers=numbers,
            cfg=cfg,
            results=results,
            series=series,
            spikes=spikes,
            n_pres=n_pres,
            time_s=time_s,
            mid_s=mid_s,
            block_s=block_s,
            blocks=blocks,
            block_mid_s=block_mid_s,
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
