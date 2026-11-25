"""One digit for ten seconds: what van Rossum plasticity does without stimulus changes.

exp001-exp005 changed the digit every 200 ms, so the direction of each weight
change kept switching with the input. Here each network sees a single MNIST
digit for the whole stream (default 50 x 200 ms = 10 s, no state reset), with
fresh Poisson spikes in every 200-ms window. Three different random digits, each
started from the trained weights, test whether the effects are digit-specific.

The sources are exp022's rate-matched TR02 pair (PING and COBA trained with a
25 Hz activity ceiling, seed 42). Each digit runs frozen and with van Rossum
STDP + synaptic scaling on the sensory weights X→E. Measured: ISIs, Eoin's
rhythmicity contrast (exp054), spectra, and the direction of weight change:
potentiation or depression, on the digit's own pixels and on the others, and as
a function of the initial weight.
"""

from snnlab import lang
from snnlab.plasticity import NonlinearSTDP, Plasticity, SynapticScaling

SLUG = "exp006"
# Numbered exp181 until 2026-11-25. Compute runs recorded under that number
# remain valid inputs; see source_experiment().
LEGACY_SLUG = "exp181"
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
DESIGN_RULES = ("frozen", "both")  # frozen first for every digit


def stream_kind(digit):
    return f"digit{digit}"


def stream_digit(kind):
    if not kind.startswith("digit") or not kind[5:].isdigit():
        raise ValueError(f"Unknown stream {kind!r}")
    return int(kind[5:])


def configuration(**overrides):
    """Fresh resolved settings; CLI overrides are validated before use.

    Units: ms, Hz and µS (the COBA conductance unit of the checkpoints).
    """
    cfg = dict(
        source_run="exp022-r039-compute",
        # TR02 cells with a 25 Hz activity ceiling: rate-matched PING and COBA.
        source_cells={
            "ping": "ping__rt25hz__seed42",
            "coba": "coba__rt25hz__seed42",
        },
        checkpoint_role="final_epoch",
        seed=42,
        device="cpu",
        threads=0,  # 0 = all available cores
        # Stimulus: one official MNIST test image (outside the 1,000-image accuracy
        # subset, so no network trained on it), repeated in consecutive 200-ms
        # windows with fresh Poisson spikes and no state reset.
        presentations=50,  # 200-ms windows: 50 = 10 s
        presentation_steps=None,  # None = the source protocol (200 ms); tests only
        digits=None,  # None = digit_count distinct random digits from digit_seed
        digit_count=3,
        digit_seed=181001,
        stream_seed=181002,  # which image of each digit (first in a seeded order)
        stream_encoder_seed=181003,
        digit_pixel_threshold=0.25,  # intensity marking the digit's own pixels
        # A synapse counts as potentiated / depressed when |ΔW| exceeds this
        # fraction of the trained mean weight.
        change_tolerance_fraction=0.01,
        tendency_bins=20,  # initial-weight bins (W0 / w_max) for ΔW against W0
        # Van Rossum STDP: constant potentiation c_p, depression c_d·w per pair.
        stdp_tau_ms=20.0,
        stdp_reference_tau_ms=20.0,  # amplitudes scale by reference/tau (equal area)
        stdp_depression_fraction=0.003,
        # Van Rossum synaptic scaling, time-compressed 100x (τ 100 s -> 1 s).
        scaling_tau_rate_ms=1000.0,
        scaling_beta=4e-3,
        scaling_gamma=1e-3,
        setpoint_policy="frozen-condition mean E rate of the same network and digit",
        # Measurement.
        block_presentations=5,
        weight_snapshot_count=6,  # full matrices at 0, 2, 4, 6, 8 and 10 s
        weight_histogram_bins=60,
        tracked_synapses=1000,
        tracked_synapse_seed=181004,
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
        "digit_count",
        "tendency_bins",
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["presentations"] < 2 * cfg["block_presentations"]:
        raise ValueError("Need at least two measurement blocks")
    if cfg["digits"] is None:
        import numpy as np

        if cfg["digit_count"] > 10:
            raise ValueError("digit_count must be at most 10")
        rng = np.random.default_rng(cfg["digit_seed"])
        cfg["digits"] = [
            int(d) for d in rng.choice(10, cfg["digit_count"], replace=False)
        ]
    digits = list(cfg["digits"])
    if (
        not digits
        or any(type(d) is not int or not 0 <= d <= 9 for d in digits)
        or len(set(digits)) != len(digits)
    ):
        raise ValueError("digits must be distinct integers 0-9")
    cfg["digits"] = digits
    if not 0 < cfg["digit_pixel_threshold"] < 1:
        raise ValueError("digit_pixel_threshold must lie in (0, 1)")
    if not 0 < cfg["change_tolerance_fraction"] < 1:
        raise ValueError("change_tolerance_fraction must lie in (0, 1)")
    if cfg["presentation_steps"] is not None and (
        type(cfg["presentation_steps"]) is not int or cfg["presentation_steps"] < 100
    ):
        raise ValueError("presentation_steps must be None or an integer >= 100")
    if type(cfg["threads"]) is not int or cfg["threads"] < 0:
        raise ValueError("threads must be a nonnegative integer (0 = all cores)")
    if cfg["device"] != "cpu":
        raise ValueError("Plastic simulation is validated on CPU only")
    return cfg


def conditions(cfg):
    """Every network x digit, frozen then STDP + scaling; each starts trained."""
    return [
        dict(
            id=f"{network}__{stream_kind(digit)}__{rule}",
            network=network,
            rule=rule,
            stream=stream_kind(digit),
            digit=digit,
            tau_stdp_ms=cfg["stdp_tau_ms"],
            setpoint=1.0,
            label=f"{NETWORK_LABELS[network]} · digit {digit} · {RULE_LABELS[rule]}",
        )
        for network in cfg["source_cells"]
        for digit in cfg["digits"]
        for rule in DESIGN_RULES
    ]


def stream_length(cfg, kind):
    stream_digit(kind)
    return cfg["presentations"]


def stream_plan(cfg, kind, pool, labels, input_rate_hz):
    """The same image in every 200-ms window: the first pool image of the digit."""
    digit = stream_digit(kind)
    image = next((int(i) for i in pool if labels[i] == digit), None)
    if image is None:
        raise ValueError(f"No image of digit {digit} in the stream pool")
    return [
        dict(image=image, label=digit, rate_hz=float(input_rate_hz))
        for _ in range(stream_length(cfg, kind))
    ]


def setpoint_window(cfg, kind):
    """Presentations whose frozen mean E rate defines the scaling setpoint."""
    return 0, stream_length(cfg, kind)


def digit_mask(image, threshold):
    """The digit's own pixels: intensity above threshold (flat, input order)."""
    import numpy as np

    return np.asarray(image, dtype=float).ravel() > threshold


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
