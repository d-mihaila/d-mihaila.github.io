"""Plasticity parameters that steer the PING rhythm: STDP window and scaling setpoint.

The source is the exp022 TR01 PING network (seed 42, no rate penalty), whose
rhythmicity contrast is moderate (about 0.36), so plasticity can sharpen or blur
it. The full van Rossum rule (STDP + synaptic scaling) runs on the sensory
weights with one parameter changed at a time:

1. The STDP window τ ∈ {2.5, 5, 10, 20, 40} ms, against a rhythm period of about
   10 ms. Amplitudes scale with 20 ms / τ, so every window has the same area and
   the uncorrelated learning speed stays fixed; only timing resolution changes.
2. The scaling setpoint ν* at 0.5× and 2× the frozen E rate (τ = 20 ms).
"""

from snnlab import lang
from snnlab.plasticity import NonlinearSTDP, Plasticity, SynapticScaling

SLUG = "exp003"
# Numbered exp178 until 2026-11-25. Compute runs recorded under that number
# remain valid inputs; see source_experiment().
LEGACY_SLUG = "exp178"
PLASTIC_PROJECTION = "input_to_E"
# Native parameter name -> exp022 checkpoint matrix (stored source-to-target).
MAPPING = {
    "input_to_E.weight": "W_ff.0",
    "readout_projection.weight": "W_ff.1",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
}
RULES = ("frozen", "stdp", "scaling", "both")
RULE_LABELS = {
    "frozen": "Frozen",
    "stdp": "STDP only",
    "scaling": "Scaling only",
    "both": "STDP + scaling",
}
NETWORK_LABELS = {"ping": "PING", "coba": "COBA"}


def configuration(**overrides):
    """Fresh resolved settings; CLI overrides are validated before use.

    Units: ms, Hz and µS (the COBA conductance unit of the checkpoints).
    """
    cfg = dict(
        source_run="exp022-r039-compute",
        # TR01 PING: no rate penalty, all of MNIST, moderate rhythmicity.
        source_cells={"ping": "ping__canonical__seed42"},
        tau_sweep_ms=[2.5, 5.0, 10.0, 20.0, 40.0],
        setpoint_sweep=[0.5, 2.0],
        checkpoint_role="final_epoch",
        seed=42,
        device="cpu",
        threads=0,  # 0 = all available cores
        # Stream: official MNIST test images outside the 1,000-image accuracy
        # subset, in a seeded order, so no network has trained on them. No state
        # reset between presentations.
        presentations=50,
        presentation_steps=None,  # None = the source protocol (200 ms); tests only
        stream_seed=178001,
        stream_encoder_seed=178002,
        # Van Rossum STDP: constant potentiation c_p, depression c_d·w per pair.
        stdp_tau_ms=20.0,
        stdp_reference_tau_ms=20.0,  # amplitudes scale by reference/tau (equal area)
        stdp_depression_fraction=0.003,
        # Van Rossum synaptic scaling, time-compressed 100x (τ 100 s -> 1 s).
        scaling_tau_rate_ms=1000.0,
        scaling_beta=4e-3,
        scaling_gamma=1e-3,
        setpoint_policy="frozen-condition mean E rate of the same network and stream",
        # Measurement.
        block_presentations=5,
        weight_snapshot_count=5,
        weight_histogram_bins=60,
        tracked_synapses=1000,
        tracked_synapse_seed=178003,
        accuracy_every=10,
        accuracy_images=500,
        accuracy_batch_size=64,
        accuracy_sample_seed=42,
        accuracy_encoder_seed=20260415,
        rhythm_lag_ms=100.0,  # exp054 rhythmicity settings
        rhythm_bin_ms=1.0,
        rhythm_burn_ms=100.0,
        rhythm_band_hz=[20.0, 200.0],
        rate_bin_ms=1.0,
        cycle_min_separation_ms=3.0,
    )
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    cfg.update(overrides)
    for key in (
        "presentations",
        "block_presentations",
        "weight_snapshot_count",
        "tracked_synapses",
        "accuracy_every",
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not cfg["tau_sweep_ms"] or any(t <= 0 for t in cfg["tau_sweep_ms"]):
        raise ValueError("tau_sweep_ms must hold positive window lengths")
    if cfg["stdp_tau_ms"] not in cfg["tau_sweep_ms"]:
        raise ValueError("The reference window must be part of the sweep")
    if any(f <= 0 or f == 1 for f in cfg["setpoint_sweep"]):
        raise ValueError("setpoint_sweep holds positive factors other than 1")
    if cfg["presentations"] < 2 * cfg["block_presentations"]:
        raise ValueError("Need at least two measurement blocks")
    if cfg["presentation_steps"] is not None and (
        type(cfg["presentation_steps"]) is not int or cfg["presentation_steps"] < 100
    ):
        raise ValueError("presentation_steps must be None or an integer >= 100")
    if (
        type(cfg["accuracy_images"]) is not int
        or not 0 <= cfg["accuracy_images"] <= 1000
    ):
        raise ValueError("accuracy_images must be an integer in [0, 1000]")
    if type(cfg["threads"]) is not int or cfg["threads"] < 0:
        raise ValueError("threads must be a nonnegative integer (0 = all cores)")
    if cfg["device"] != "cpu":
        raise ValueError("Plastic simulation is validated on CPU only")
    return cfg


def tag(value):
    return f"{value:g}".replace(".", "p")


def conditions(cfg):
    """Frozen, then the τ sweep (setpoint 1), then the setpoint sweep (τ = 20 ms)."""
    base = dict(network="ping", stream="balanced")
    rows = [
        dict(
            base,
            id="frozen",
            rule="frozen",
            tau_stdp_ms=cfg["stdp_tau_ms"],
            setpoint=1.0,
            sweep=None,
        )
    ]
    rows += [
        dict(
            base,
            id=f"tau{tag(tau)}",
            rule="both",
            tau_stdp_ms=tau,
            setpoint=1.0,
            sweep="tau",
        )
        for tau in cfg["tau_sweep_ms"]
    ]
    rows += [
        dict(
            base,
            id=f"setpoint{tag(f)}",
            rule="both",
            tau_stdp_ms=cfg["stdp_tau_ms"],
            setpoint=f,
            sweep="setpoint",
        )
        for f in cfg["setpoint_sweep"]
    ]
    for row in rows:
        row["label"] = label(row)
    return rows


def label(c):
    if c["rule"] == "frozen":
        return "Frozen"
    if c["sweep"] == "setpoint":
        return f"ν* ×{c['setpoint']:g}"
    return f"τ {c['tau_stdp_ms']:g} ms"


def stream_plan(cfg, kind, pool, labels, input_rate_hz):
    """Image index and peak input rate (Hz) for every presentation of a stream."""
    if kind != "balanced":
        raise ValueError(f"Unknown stream {kind!r}")
    order = pool[: cfg["presentations"]]
    return [
        dict(image=int(i), label=int(labels[i]), rate_hz=input_rate_hz) for i in order
    ]


def setpoint_window(cfg, kind):
    """Presentations whose frozen mean E rate defines the scaling setpoint."""
    return 0, cfg["presentations"]


def rule_parameters(cfg, initial_weights, tau_ms):
    """Map van Rossum's rule onto SNNLab's NonlinearSTDP for one network.

    Each pre-before-post pair adds c_p; each post-before-pre pair removes c_d·w.
    NonlinearSTDP with mu_plus = 0 and mu_minus = 1 gives this when
    a_minus = c_d·w_max and a_plus = c_p. With equal time constants the
    uncorrelated fixed point is w* = c_p / c_d, which we put at the trained mean.
    Amplitudes scale by reference_tau / tau, so the window area (and therefore
    the uncorrelated learning speed) does not change with tau.
    """
    import numpy as np

    values = np.asarray(initial_weights, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or values.min() < 0:
        raise ValueError("Initial weights must be a finite nonnegative matrix")
    if not tau_ms > 0:
        raise ValueError("tau_ms must be positive")
    w_max = float(np.asarray(initial_weights).max())
    mean = float(values.mean())
    c_d = cfg["stdp_depression_fraction"]
    area = cfg["stdp_reference_tau_ms"] / tau_ms
    return dict(
        w_min_us=0.0,
        w_max_us=w_max,
        initial_mean_us=mean,
        equilibrium_us=mean,
        tau_ms=tau_ms,
        area_factor=area,
        a_plus_us=c_d * mean * area,
        a_minus_us=c_d * w_max * area,
        c_d=c_d,
    )


def plasticity_request(cfg, rule, params, target_rate_hz):
    """None for frozen; otherwise STDP, scaling or both on the sensory projection."""
    if rule == "frozen":
        return None
    if rule not in RULES:
        raise ValueError(f"Unknown rule {rule!r}")
    bounds = dict(
        w_min=params["w_min_us"] * lang.uS, w_max=params["w_max_us"] * lang.uS
    )
    rules, scaling = {}, {}
    if rule in ("stdp", "both"):
        rules[PLASTIC_PROJECTION] = NonlinearSTDP(
            a_plus=params["a_plus_us"] * lang.uS,
            a_minus=params["a_minus_us"] * lang.uS,
            tau_pre=params["tau_ms"] * lang.ms,
            tau_post=params["tau_ms"] * lang.ms,
            mu_plus=0.0,
            mu_minus=1.0,
            **bounds,
        )
    if rule in ("scaling", "both"):
        if not target_rate_hz > 0:
            raise ValueError("The scaling target rate must be positive")
        scaling[PLASTIC_PROJECTION] = SynapticScaling(
            target_rate=target_rate_hz * lang.Hz,
            tau_rate=cfg["scaling_tau_rate_ms"] * lang.ms,
            beta=cfg["scaling_beta"],
            gamma=cfg["scaling_gamma"],
            **bounds,
        )
    return Plasticity(rules=rules, scaling=scaling)


def native_weights(weights, architecture):
    names = list(MAPPING) if architecture == "ping" else list(MAPPING)[:2]
    return {name: weights[MAPPING[name]].clone() for name in names}


def populations(architecture):
    return ("E", "I") if architecture == "ping" else ("E",)


def snapshot_indices(presentations, count):
    """Presentation boundaries (0 = before learning) at which full matrices are kept."""
    import numpy as np

    return sorted({int(round(x)) for x in np.linspace(0, presentations, count)})


def accuracy_boundaries(presentations, every):
    """Presentation boundaries (0 = trained weights) with a held-out evaluation."""
    if type(every) is not int or every < 1:
        raise ValueError("accuracy_every must be a positive integer")
    return sorted({0, presentations, *range(every, presentations + 1, every)})


def tracked_synapses(shape, count, seed):
    """Flat indices of a fixed random sample of synapses, sorted, without replacement."""
    import numpy as np

    total = int(np.prod(shape))
    if not 1 <= count <= total:
        raise ValueError("tracked_synapses must lie between 1 and the synapse count")
    return np.sort(np.random.default_rng(seed).choice(total, count, replace=False))


def test_sample_indices(*, total_images=10000, samples=1000, seed=42):
    """The manuscript's ordered official-test subset (exp173); smaller sizes nest."""
    import numpy as np

    if type(samples) is not int or not 1 <= samples <= total_images:
        raise ValueError("Invalid test subset size")
    return np.random.RandomState(seed).choice(total_images, samples, replace=False)


def stream_pool(total_images, seed):
    """Official-test indices outside the 1,000-image accuracy subset, seeded order."""
    import numpy as np

    held_out = test_sample_indices(total_images=total_images, samples=1000)
    pool = np.setdiff1d(np.arange(total_images), held_out)
    return np.random.default_rng(seed).permutation(pool)


def author_network(cfg, network):
    """Native network. PING: X→E⇄I, E→R. COBA: X→E, E→R."""
    spec = cfg["networks"][network]
    net = lang.Network(f"{SLUG}_{network}", dt=cfg["dt_ms"] * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", cfg["n_input"]),
        signal_type="spikes",
        unit="spike",
    )
    pops = {}
    for name in populations(spec["architecture"]):
        neuron = dict(cfg["neurons"][name])
        neuron["tau_mem"] = neuron.pop("tau_mem_ms") * lang.ms
        neuron["refractory"] = neuron.pop("refractory_ms") * lang.ms
        size = cfg["n_e"] if name == "E" else cfg["n_i"]
        pops[name] = net.population(name, size=size, neuron=lang.COBA_LIF(**neuron))

    def connect(name, src, dst, tau, inhibitory=False, recurrent=False):
        net.connect(
            src,
            dst,
            name=name,
            synapse=(lang.GABA if inhibitory else lang.AMPA)(tau=tau * lang.ms),
            weight=lang.Constant(0),
            initialization_scaling="direct",
            constraint=lang.NonNegative(),
            delay=(cfg["dt_ms"] if recurrent else 0) * lang.ms,
            **({"connection": "recurrent"} if recurrent else {}),
        )

    connect("input_to_E", drive, pops["E"].excitatory, cfg["tau_ampa_ms"])
    if spec["architecture"] == "ping":
        connect(
            "E_to_I",
            pops["E"].spikes,
            pops["I"].excitatory,
            cfg["tau_ampa_ms"],
            recurrent=True,
        )
        connect(
            "I_to_E",
            pops["I"].spikes,
            pops["E"].inhibitory,
            spec["tau_gaba_ms"],
            inhibitory=True,
            recurrent=True,
        )
    readout = net.population(
        "readout",
        size=cfg["n_output"],
        spiking=True,
        neuron=lang.LeakyIntegrator(
            tau=cfg["readout_tau_ms"] * lang.ms,
            soft_reset_threshold=cfg["readout_threshold"],
            initial_voltage=0,
            surrogate_slope=cfg["surrogate_slope"],
        ),
    )
    net.connect(
        pops["E"].spikes,
        readout.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=cfg["readout_tau_ms"] * lang.ms),
        weight=lang.Constant(0),
        initialization_scaling="direct",
        constraint=lang.NonNegative(),
    )
    net.output(
        "class_scores",
        lang.reduce(
            readout.pre_reset_voltage,
            operation="mean",
            over="time",
            name="mean_class_scores",
        ),
    )
    for name in pops:
        net.output(
            name + "_count",
            lang.reduce(
                pops[name].spikes, operation="sum", over="time", name=name + "_counts"
            ),
        )
    return lang.compile(net)


def source_experiment(run_id):
    """Experiment ID recorded in a compute run, including runs from before renumbering."""
    return LEGACY_SLUG if run_id.startswith(f"{LEGACY_SLUG}-") else SLUG
