"""Attended image reconstruction through a fixed contrast-attention circuit."""

import math

import torch
from experiments.exp175.custom import reconstruction  # noqa: F401
from snnlab import lang

SLUG = "exp175"
TRAINABLE = ("shared_encoder.weight", "readout_projection.weight")
LOCAL_ROLES = ("E_to_E", "E_to_I", "I_to_E", "I_to_I")
DISABLED = tuple(f"{s}_{p}_to_{p}" for s in "ABC" for p in "EI") + ("background_C",)
FROZEN = (
    tuple(f"{s}_{role}.weight" for s in "ABC" for role in LOCAL_ROLES)
    + tuple(f"{s}_E_to_C_{p}.weight" for s in "AB" for p in "EI")
    + ("background_C.weight",)
)


def configuration(**overrides):
    """Counts refer to unique images; cue contrast is a fixed mean-matched ratio."""
    cfg = dict(
        architecture="relay",
        source_run="exp175-r001-import",
        tr02_run="exp022-r039-compute",
        source_cell="ping__off__seed42",
        train_images=1000,
        validation_images=100,
        split_seed=42,
        seed=175001,
        epochs=10,
        batch_size=64,
        microbatch_size=64,
        validation_draws=3,
        lr=0.0004,
        weight_decay=0.0,
        grad_clip=1.0,
        dt_ms=0.1,
        duration_ms=200.0,
        switch_ms=100.0,
        windows_ms=((50.0, 90.0), (150.0, 190.0)),
        contrast_ratio=12.0,
        mean_input_hz=35.44717887154862,
        n_input=784,
        n_e=1024,
        n_i=256,
        n_output=784,
        decoder_init_mean=0.05,
        decoder_init_std=0.04,
        white_pixel_rate_hz=500.0,
        readout_tau_ms=2.0,
        readout_threshold=1.0,
        target_image_file=None,
        surrogate_slope=1.0,
        voltage_grad_dampen=1000.0,
        tau_ampa_ms=2.0,
        tau_gaba_ms=6.0,
        relay_sigma=0.00025,
        relay_e_total_us=0.05,
        relay_i_total_us=0.005,
        relay_tau_ms=6.0,
        relay_delay_ms=2.0,
        device="mps",
        mps_memory_fraction=1.0,
        threads=1,
        compile_mode="default",
        precision="highest",
    )
    if unknown := set(overrides) - set(cfg):
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    cfg.update(overrides)
    for key in (
        "train_images",
        "validation_images",
        "split_seed",
        "seed",
        "epochs",
        "batch_size",
        "microbatch_size",
        "validation_draws",
        "threads",
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (
        not (
            10 <= cfg["train_images"] <= 6300 and 10 <= cfg["validation_images"] <= 700
        )
        or cfg["split_seed"] != 42
    ):
        raise ValueError("Use subsets of TR02's 6,300/700 partitions, seed 42")
    if cfg["microbatch_size"] > cfg["batch_size"]:
        raise ValueError("microbatch_size exceeds the effective batch size")
    for key in (
        "lr",
        "grad_clip",
        "dt_ms",
        "duration_ms",
        "switch_ms",
        "mean_input_hz",
        "contrast_ratio",
        "surrogate_slope",
        "voltage_grad_dampen",
    ):
        if isinstance(cfg[key], bool) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if cfg["contrast_ratio"] <= 1 or cfg["weight_decay"] != 0:
        raise ValueError("Require contrast > 1 and TR02's zero weight decay")
    if (cfg["n_input"], cfg["n_e"], cfg["n_i"], cfg["n_output"]) != (
        784,
        1024,
        256,
        784,
    ):
        raise ValueError("Pinned architecture dimensions cannot change")
    if cfg["dt_ms"] != 0.1 or cfg["voltage_grad_dampen"] != 1000.0:
        raise ValueError("Preserve the frozen body's timestep and voltage derivative")
    if cfg["compile_mode"] not in (None, "default"):
        raise ValueError("Unsupported native compile mode")
    if cfg["precision"] not in ("highest", "auto"):
        raise ValueError("Unsupported precision")
    validate_timing(cfg)
    cfg["steps"] = round(cfg["duration_ms"] / cfg["dt_ms"])
    cfg["switch_step"] = round(cfg["switch_ms"] / cfg["dt_ms"])
    cfg["window_steps"] = [
        [round(t / cfg["dt_ms"]) for t in w] for w in cfg["windows_ms"]
    ]
    window_lengths = [hi - lo for lo, hi in cfg["window_steps"]]
    if window_lengths[0] != window_lengths[1]:
        raise ValueError("Spike-count decoding requires equal scoring-window durations")
    if (
        cfg["white_pixel_rate_hz"] <= 0
        or cfg["white_pixel_rate_hz"] > 1000 / cfg["dt_ms"]
    ):
        raise ValueError(
            "White-pixel rate must be positive and at most one spike per timestep"
        )
    cfg["white_pixel_count"] = (
        cfg["white_pixel_rate_hz"] * window_lengths[0] * cfg["dt_ms"] / 1000
    )
    cfg["validation_presentations"] = 4 * cfg["validation_images"]
    return cfg


def validate_timing(cfg):
    if cfg["architecture"] not in ("relay", "single"):
        raise ValueError("Architecture must be relay or single")
    if not 0 < cfg["switch_ms"] < cfg["duration_ms"]:
        raise ValueError("Switch must occur inside the presentation")
    if len(cfg["windows_ms"]) != 2:
        raise ValueError("Require one scoring window per cue phase")
    for bounds, (start, stop) in zip(
        ((0, cfg["switch_ms"]), (cfg["switch_ms"], cfg["duration_ms"])),
        cfg["windows_ms"],
        strict=True,
    ):
        if not bounds[0] <= start < stop <= bounds[1]:
            raise ValueError("Scoring windows must be inside their cue phases")
    for t in (
        cfg["duration_ms"],
        cfg["switch_ms"],
        *(v for w in cfg["windows_ms"] for v in w),
    ):
        if not math.isclose(t / cfg["dt_ms"], round(t / cfg["dt_ms"])):
            raise ValueError("Timing must align with the timestep")


def author_network(cfg):
    """Compile the native circuit; no build, input generation or execution."""
    net = lang.Network("mnist_fixed_contrast_finetuning", dt=cfg["dt_ms"] * lang.ms)
    encoder = net.parameter(
        "shared_encoder.weight",
        shape=(1024, 784),
        initializer=lang.Constant(0),
        unit="uS",
        constraint=lang.NonNegative(),
        initialization_scaling="direct",
    )
    senders = "A" if cfg["architecture"] == "single" else "ABC"
    pops = {}
    for sender in senders:
        pops[sender] = {}
        for kind, size, cap, leak, refractory in (
            ("E", 1024, 1.0, 0.05, 1.2),
            ("I", 256, 0.5, 0.1, 0.6),
        ):
            pops[sender][kind] = net.population(
                f"{sender}_{kind}",
                size=size,
                neuron=lang.COBA_LIF(
                    tau_mem=cap / leak * lang.ms,
                    capacitance_nf=cap,
                    leak_us=leak,
                    resting_mv=-65.0,
                    threshold_mv=-50.0,
                    reset_mv=-65.0,
                    initial_voltage_mv=-65.0,
                    refractory=refractory * lang.ms,
                    voltage_grad_dampen=cfg["voltage_grad_dampen"],
                ),
            )

    def connect(source, target, name, tau, *, delay=0.0, gaba=False, weight=None):
        net.connect(
            source,
            target,
            name=name,
            synapse=(lang.GABA if gaba else lang.AMPA)(tau=tau * lang.ms),
            weight=lang.Constant(0.0) if weight is None else weight,
            constraint=lang.NonNegative(),
            initialization_scaling="direct",
            delay=delay * lang.ms,
            enabled=name not in DISABLED,
        )

    for sender in senders:
        for source, target, tau in (
            ("E", "E", cfg["tau_ampa_ms"]),
            ("E", "I", cfg["tau_ampa_ms"]),
            ("I", "E", cfg["tau_gaba_ms"]),
            ("I", "I", cfg["tau_gaba_ms"]),
        ):
            connect(
                pops[sender][source].spikes,
                pops[sender][target].inhibitory
                if source == "I"
                else pops[sender][target].excitatory,
                f"{sender}_{source}_to_{target}",
                tau,
                delay=cfg["dt_ms"],
                gaba=source == "I",
            )
    for sender in "A" if cfg["architecture"] == "single" else "AB":
        drive = net.input(
            f"X_{sender}",
            shape=("time", "batch", 784),
            signal_type="spikes",
            unit="spike",
        )
        connect(
            drive,
            pops[sender]["E"].excitatory,
            f"sensory_{sender}",
            cfg["tau_ampa_ms"],
            weight=encoder,
        )
        for kind in "" if cfg["architecture"] == "single" else "EI":
            connect(
                pops[sender]["E"].spikes,
                pops["C"][kind].excitatory,
                f"{sender}_E_to_C_{kind}",
                cfg["relay_tau_ms"],
                delay=cfg["relay_delay_ms"],
            )
    if cfg["architecture"] == "relay":
        background = net.input(
            "X_C", shape=("time", "batch", 1024), signal_type="spikes", unit="spike"
        )
        connect(background, pops["C"]["E"].excitatory, "background_C", 2.0)
    mask = net.input(
        "score_window", shape=("time", "batch"), signal_type="mask", unit="1"
    )
    output = net.population(
        "readout",
        size=cfg["n_output"],
        spiking=True,
        neuron=lang.LeakyIntegrator(
            tau=cfg["readout_tau_ms"] * lang.ms,
            soft_reset_threshold=cfg["readout_threshold"],
            initial_voltage=0.0,
            surrogate_slope=cfg["surrogate_slope"],
        ),
    )
    net.connect(
        pops["A" if cfg["architecture"] == "single" else "C"]["E"].spikes,
        output.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=cfg["readout_tau_ms"] * lang.ms),
        weight=lang.Constant(0.0),
        constraint=lang.NonNegative(),
        initialization_scaling="direct",
    )
    counts = lang.reduce(
        output.spikes,
        operation="sum",
        over="time",
        name="readout_counts",
        mask=mask,
    )
    pixels = lang.custom_operation(
        "exp175.pixel_output/v1",
        counts,
        name="pixel_intensities",
        shape=("batch", 784),
        unit="1",
        white_count=cfg["white_pixel_count"],
    )
    net.output("pixels", pixels)
    net.output("counts", counts)
    training = lang.TrainSpec(
        objectives=(
            lang.CustomObjective(
                "exp175.pixel_mse/v1",
                prediction=pixels,
                target="image_index",
                image_file=cfg["target_image_file"],
            ),
        ),
        parameter_groups=(
            lang.ParameterGroup(TRAINABLE, name="encoder_decoder", lr=cfg["lr"]),
            lang.ParameterGroup(
                frozen_parameters(cfg), name="fixed_attention", lr=0.0, frozen=True
            ),
        ),
        optimizer=lang.AdamW(lr=cfg["lr"], weight_decay=cfg["weight_decay"]),
        gradient_clip=cfg["grad_clip"],
        surrogate=lang.FastSigmoid(slope=cfg["surrogate_slope"]),
        epochs=cfg["epochs"],
        presentation_duration=cfg["duration_ms"] * lang.ms,
    )
    return lang.compile(net, training=training, target=None)


def initialise(raw, cfg):
    """Keep the TR02 encoder; initialise a nonnegative spiking pixel decoder."""
    parameters = {
        "shared_encoder.weight": raw["W_ff.0"].clone(),
        "readout_projection.weight": (
            cfg["decoder_init_mean"]
            + torch.randn(
                cfg["n_e"],
                cfg["n_output"],
                generator=torch.Generator().manual_seed(cfg["seed"]),
            )
            * cfg["decoder_init_std"]
        ).clamp_min(0),
    }
    local = {
        "W_ee.1": torch.zeros(cfg["n_e"], cfg["n_e"]),
        "W_ii.1": torch.zeros(cfg["n_i"], cfg["n_i"]),
        **raw,
    }
    for sender in "ABC":
        for role, key in zip(
            LOCAL_ROLES, ("W_ee.1", "W_ei.1", "W_ie.1", "W_ii.1"), strict=True
        ):
            parameters[f"{sender}_{role}.weight"] = local[key].clone()
    coordinate = torch.arange(cfg["n_e"], dtype=torch.float64) / cfg["n_e"]
    distance = (coordinate[:, None] - coordinate[None, :]).abs()
    distance = torch.minimum(distance, 1 - distance)
    kernel = torch.exp(-0.5 * (distance / cfg["relay_sigma"]) ** 2)
    kernel /= kernel.sum(0, keepdim=True)
    for sender in "AB":
        parameters[f"{sender}_E_to_C_E.weight"] = (
            kernel * cfg["relay_e_total_us"]
        ).float()
        parameters[f"{sender}_E_to_C_I.weight"] = torch.full(
            (cfg["n_e"], cfg["n_i"]), cfg["relay_i_total_us"] / cfg["n_e"]
        )
    parameters["background_C.weight"] = torch.zeros(cfg["n_e"], cfg["n_e"])
    for name in DISABLED:
        if parameters[name + ".weight"].count_nonzero():
            raise ValueError(
                f"Disabled projection must have frozen zero weights: {name}"
            )
    return {
        name: value
        for name, value in parameters.items()
        if name in (*TRAINABLE, *frozen_parameters(cfg))
    }


def frozen_parameters(cfg):
    """The single-network protocol contains only A's fixed PING feedback."""
    if cfg["architecture"] == "single":
        return tuple(f"A_{role}.weight" for role in LOCAL_ROLES)
    return FROZEN
