"""Van Rossum plasticity on the sensory weights of trained TR01 PING and COBA networks.

Settings, the van Rossum rule and the native SNNLab networks live here. The
source networks are the exp022 TR01 (canonical, no rate penalty) cells for seed
42: PING has a trained E→I→E loop; COBA has the same E population and readout,
with its E–I weights exactly zero, so it is authored without an I population.
"""

from snnlab import lang
from snnlab.plasticity import NonlinearSTDP, Plasticity, SynapticScaling

SLUG = "exp176"
NETWORKS = ("ping", "coba")
CONDITIONS = ("frozen", "van_rossum")
PLASTIC_PROJECTION = "input_to_E"
# Native parameter name -> exp022 checkpoint matrix (stored source-to-target).
MAPPING = {
    "input_to_E.weight": "W_ff.0",
    "readout_projection.weight": "W_ff.1",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
}
MAPPING_COBA = ("input_to_E.weight", "readout_projection.weight")
LABELS = {
    "ping": "PING",
    "coba": "COBA",
    "frozen": "Frozen",
    "van_rossum": "Van Rossum",
}


def configuration(**overrides):
    """Fresh resolved settings. CLI overrides are validated before use.

    Units: times in ms, rates in Hz, weights in µS (the COBA conductance unit
    of the source checkpoints). The rule constants follow van Rossum, Bi &
    Turrigiano (2000); see `rule_parameters` for how they become SNNLab values.
    """
    cfg = dict(
        # Source: exp022 TR01 "canonical" cells = no rate penalty, all MNIST.
        source_run="exp022-r039-compute",
        source_cells={
            "ping": "ping__canonical__seed42",
            "coba": "coba__canonical__seed42",
        },
        training_family="TR-01",
        checkpoint_role="final_epoch",
        seed=42,
        device="cpu",
        threads=0,  # 0 = all available cores (plastic steps are matrix-wide)
        # Stream: consecutive MNIST presentations with no state reset between
        # them, so plasticity acts on one continuous trajectory.
        presentations=50,
        stream_partition="source validation split (images never used for gradient updates)",
        stream_order="first N images of the source validation split, stored order",
        stream_encoder_seed=176001,
        state_policy="continuous: neuronal, synaptic and plastic state carry across presentations",
        # Van Rossum STDP (mixed rule): constant potentiation, depression ∝ w.
        plastic_projection=PLASTIC_PROJECTION,
        stdp_tau_ms=20.0,
        stdp_depression_fraction=0.003,
        stdp_equilibrium="initial mean weight",
        w_max_policy="largest initial weight of each network",
        # Van Rossum synaptic scaling (PI control of each E neuron's rate).
        scaling_target_policy="frozen-condition mean E rate of the same network and stream",
        scaling_tau_rate_ms=1000.0,
        scaling_beta=4e-3,
        scaling_gamma=1e-3,
        time_compression=100.0,
        # Evidence and measurement.
        weight_snapshot_count=5,
        weight_histogram_bins=60,
        isi_block_presentations=5,  # ISI, CV and spectra per 1-s block
        rate_bin_ms=1.0,
        # Spectral band for the population rhythm. These trained PING networks
        # oscillate faster than classic 30–80 Hz gamma, so the band is wide.
        rhythm_band_hz=[20.0, 200.0],
        receptive_fields=4,
        # Individual synapses followed after every presentation (uniform sample
        # without replacement over all 784 x 1024 X->E synapses).
        tracked_synapses=1000,
        tracked_synapse_seed=176002,
        # Held-out test accuracy with the current weights every N presentations.
        accuracy_every=5,
        accuracy_images=1000,
        accuracy_batch_size=64,
        accuracy_sample_seed=42,
        accuracy_encoder_seed=20260415,
        accuracy_readout="argmax of trial-mean pre-reset readout voltage",
        accuracy_partition="official MNIST test partition; uniform without replacement, no stratification",
    )
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    cfg.update(overrides)
    for key in (
        "presentations",
        "isi_block_presentations",
        "weight_snapshot_count",
        "tracked_synapses",
        "accuracy_every",
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["presentations"] < 2 * cfg["isi_block_presentations"]:
        raise ValueError("Need at least two ISI blocks of presentations")
    if (
        type(cfg["accuracy_images"]) is not int
        or not 0 <= cfg["accuracy_images"] <= 10000
    ):
        raise ValueError("accuracy_images must be an integer in [0, 10000]")
    if not 0 < cfg["stdp_depression_fraction"] < 1:
        raise ValueError("stdp_depression_fraction must lie in (0, 1)")
    if cfg["scaling_beta"] < 0 or cfg["scaling_gamma"] < 0:
        raise ValueError("Scaling gains must be nonnegative")
    if type(cfg["threads"]) is not int or cfg["threads"] < 0:
        raise ValueError("threads must be a nonnegative integer (0 = all cores)")
    if cfg["device"] != "cpu":
        raise ValueError("Plastic simulation is validated on CPU only")
    return cfg


def rule_parameters(cfg, initial_weights):
    """Map van Rossum's rule onto SNNLab's NonlinearSTDP for one network.

    van Rossum et al. (2000): each pre-before-post pair adds a constant c_p;
    each post-before-pre pair removes c_d·w. SNNLab's NonlinearSTDP with
    mu_plus = 0 and mu_minus = 1 gives exactly that when
    a_minus = c_d·(w_max − w_min) and a_plus = c_p. For uncorrelated pre/post
    firing and equal time constants the weights drift to w* = c_p / c_d. We
    put w* at the network's mean trained weight, so the rule only moves the
    mean through correlations, not through a built-in drift.
    """
    import numpy as np

    values = np.asarray(initial_weights, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or values.min() < 0:
        raise ValueError("Initial weights must be a finite nonnegative matrix")
    w_max = float(np.asarray(initial_weights).max())
    mean = float(values.mean())
    c_d = cfg["stdp_depression_fraction"]
    return dict(
        w_min_us=0.0,
        w_max_us=w_max,
        initial_mean_us=mean,
        a_plus_us=c_d * mean,
        a_minus_us=c_d * w_max,
        equilibrium_us=mean,
        c_p_us=c_d * mean,
        c_d=c_d,
    )


def plasticity_request(cfg, rule, target_rate_hz):
    """Van Rossum STDP plus synaptic scaling on the sensory projection only."""
    if not target_rate_hz > 0:
        raise ValueError("The scaling target rate must be positive")
    bounds = dict(w_min=rule["w_min_us"] * lang.uS, w_max=rule["w_max_us"] * lang.uS)
    stdp = NonlinearSTDP(
        a_plus=rule["a_plus_us"] * lang.uS,
        a_minus=rule["a_minus_us"] * lang.uS,
        tau_pre=cfg["stdp_tau_ms"] * lang.ms,
        tau_post=cfg["stdp_tau_ms"] * lang.ms,
        mu_plus=0.0,
        mu_minus=1.0,
        **bounds,
    )
    scaling = SynapticScaling(
        target_rate=target_rate_hz * lang.Hz,
        tau_rate=cfg["scaling_tau_rate_ms"] * lang.ms,
        beta=cfg["scaling_beta"],
        gamma=cfg["scaling_gamma"],
        **bounds,
    )
    return Plasticity(
        rules={PLASTIC_PROJECTION: stdp}, scaling={PLASTIC_PROJECTION: scaling}
    )


def native_weights(weights, network):
    """Checkpoint matrices for one network, keyed by native parameter name."""
    names = list(MAPPING) if network == "ping" else MAPPING_COBA
    return {name: weights[MAPPING[name]].clone() for name in names}


def populations(network):
    return ("E", "I") if network == "ping" else ("E",)


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


def snapshot_indices(presentations, count):
    """Presentation boundaries (0 = before learning) at which full matrices are kept."""
    import numpy as np

    return sorted({int(round(x)) for x in np.linspace(0, presentations, count)})


def test_sample_indices(*, total_images=10000, samples=1000, seed=42):
    """The manuscript's ordered, unstratified official-test subset (as exp173)."""
    import numpy as np

    if type(samples) is not int or not 1 <= samples <= total_images:
        raise ValueError("Invalid test subset size")
    return np.random.RandomState(seed).choice(total_images, samples, replace=False)


def author_network(cfg, network):
    """Compile the native network. PING: X→E⇄I, E→R. COBA: X→E, E→R."""
    if network not in NETWORKS:
        raise ValueError(f"Unknown network {network!r}")
    net = lang.Network(f"exp176_{network}", dt=cfg["dt_ms"] * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", cfg["n_input"]),
        signal_type="spikes",
        unit="spike",
    )
    pops = {}
    for name in populations(network):
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
    if network == "ping":
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
            cfg["tau_gaba_ms"],
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
