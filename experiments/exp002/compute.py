"""Stream MNIST through trained exp022 networks under each plasticity condition.

Every condition is one continuous trajectory (no state reset). Records every E/I
spike, the readout's score for every stream digit, weight summaries and tracked
synapses after every presentation, full matrices at a few boundaries, and
held-out test accuracy every `accuracy_every` presentations.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
import torch
from experiments.exp002 import recipe
from pingstore.library import file_sha256, source_run, stage_run, write_json_atomic
from snnlab import sim

SHAPES = {
    "W_ff.0": (784, 1024),
    "W_ff.1": (1024, 10),
    "W_ei.1": (1024, 256),
    "W_ie.1": (256, 1024),
    "W_ee.1": (1024, 1024),
    "W_ii.1": (256, 256),
}
PERFORMANCE = dict(compile_mode=None, precision="highest", projection_mode="fast")
SHARED = ("dt", "t_ms", "input_rate", "tau_ampa_ms")


def load_cell(source, cfg, cell):
    """Validate one pinned exp022 cell; return its config, pin, timing and weights."""
    conf = json.loads(source.file(cell, "config.json").read_text())
    provenance = source.details["bank_reuse"]["diagnostic_provenance"][cell]
    pin = provenance["checkpoint"]
    path = source.file(cell, pin["filename"])
    if (
        pin["role"] != cfg["checkpoint_role"]
        or pin["epoch"] != conf["epochs"]
        or file_sha256(path) != pin["sha256"]
    ):
        raise ValueError(f"Source checkpoint provenance mismatch: {cell}")
    expected = dict(
        seed=cfg["seed"],
        n_hidden=1024,
        n_inh=256,
        n_in=784,
        n_out=10,
        readout_mode="mem-mean",
        dales_law=True,
    )
    if any(conf.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{cell} does not have the expected architecture")
    if conf.get("ei_strength") not in (0.0, 1.0):
        raise ValueError(f"{cell}: unsupported E-I strength")
    unsupported = (
        "adaptive_threshold",
        "train_leak",
        "signed_readout",
        "readout_bias",
        "state_clamp",
    )
    if any(conf.get(k, False) for k in unsupported):
        raise ValueError(f"Unsupported source dynamics: {cell}")
    diagnostic = provenance["execution_records"]["config.json"]
    if diagnostic["refractory_policy"] != "exact" or diagnostic["dt"] != conf["dt"]:
        raise ValueError(f"Unsupported diagnostic timing: {cell}")
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if set(weights) != set(SHAPES):
        raise ValueError(f"Unexpected checkpoint matrices: {cell}")
    for key, shape in SHAPES.items():
        value = weights[key]
        if (
            tuple(value.shape) != shape
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid weights {key}: {cell}")
    architecture = "ping" if conf["ei_strength"] == 1.0 else "coba"
    silent = ("W_ee.1", "W_ii.1") + (
        () if architecture == "ping" else ("W_ei.1", "W_ie.1")
    )
    if any(weights[key].count_nonzero() for key in silent):
        raise ValueError(f"Unexpected nonzero recurrence: {cell}")
    spec = dict(
        cell=cell,
        architecture=architecture,
        training_run_id=conf["training_run_id"],
        tau_gaba_ms=conf["tau_gaba_ms"],
        rate_ceiling_hz=conf.get("fr_reg_upper_target_hz"),
        rate_penalty=conf.get("fr_reg_upper_strength"),
        training_images=conf["max_samples"],
        checkpoint=pin,
    )
    return conf, diagnostic, spec, {k: v.clamp_min(0) for k, v in weights.items()}


def load_source(cfg):
    """Validate every pinned source cell and record the resolved protocol."""
    source = source_run(
        ROOT / ".pingstore",
        cfg["source_run"],
        experiment="exp022",
        stage="compute",
        reference={"run_id": cfg["source_run"]},
    )
    cfg["networks"], weights, reference = {}, {}, None
    for network, cell in cfg["source_cells"].items():
        conf, diagnostic, spec, matrices = load_cell(source, cfg, cell)
        timing = (diagnostic["refractory_e_ms"], diagnostic["refractory_i_ms"])
        key = ({k: conf[k] for k in SHARED}, timing, diagnostic["duration_steps"])
        if reference is None:
            reference = (conf, diagnostic, key)
        elif key != reference[2]:
            raise ValueError(
                f"{cell} differs from the other sources in shared protocol"
            )
        cfg["networks"][network] = spec
        weights[network] = matrices
    conf, diagnostic, _ = reference
    steps = cfg["presentation_steps"] or diagnostic["duration_steps"]
    cfg.update(
        dt_ms=conf["dt"],
        steps=steps,
        presentation_ms=steps * conf["dt"],
        source_presentation_ms=diagnostic["realized_duration_ms"],
        n_input=conf["n_in"],
        n_e=conf["n_hidden"],
        n_i=conf["n_inh"],
        n_output=conf["n_out"],
        input_rate_hz=conf["input_rate"],
        tau_ampa_ms=conf["tau_ampa_ms"],
        readout_tau_ms=2.0,
        readout_threshold=1.0,
        surrogate_slope=conf["surrogate_slope"],
        dynamics_provenance="exp022 retained diagnostic exact refractory settings; canonical COBA-LIF/readout biophysics",
    )
    cfg["neurons"] = {}
    for name, c, leak, ref_key in (
        ("E", 1.0, 0.05, "refractory_e_ms"),
        ("I", 0.5, 0.1, "refractory_i_ms"),
    ):
        cfg["neurons"][name] = dict(
            tau_mem_ms=c / leak,
            capacitance_nf=c,
            leak_us=leak,
            resting_mv=-65.0,
            threshold_mv=-50.0,
            reset_mv=-65.0,
            initial_voltage_mv=-65.0,
            refractory_ms=diagnostic[ref_key],
        )
    return source, weights


def write_checkpoint(bundle, parameters, path, performance):
    built = sim.build(bundle=bundle, device="cpu", seed=42, performance=performance)
    with torch.no_grad():
        for name, value in parameters.items():
            parameter = built.model.parameter_map()[name]
            if parameter.shape != value.shape:
                raise ValueError(f"Unexpected native parameter orientation: {name}")
            parameter.copy_(value)
    torch.save(built.model.state_dict(), path)


def encode(cfg, pixels, item, index):
    """One presentation's Poisson drive; seeded per index, identical across conditions."""
    generator = torch.Generator().manual_seed(cfg["stream_encoder_seed"] + index)
    return sim.encode_images_poisson(
        torch.from_numpy(pixels[item["image"] : item["image"] + 1]),
        cfg["steps"],
        cfg["dt_ms"],
        item["rate_hz"],
        generator=generator,
    )


def weight_summary(weights, initial, bins):
    """Per-boundary statistics of the plastic matrix (source x target)."""
    flat, start = weights.ravel(), initial.ravel()
    change = flat - start
    centred = flat - flat.mean()
    sd = flat.std()
    return dict(
        mean=float(flat.mean()),
        std=float(sd),
        skew=float((centred**3).mean() / sd**3) if sd > 0 else 0.0,
        quantiles=np.quantile(flat, [0.1, 0.5, 0.9, 0.99]).tolist(),
        mean_abs_change=float(np.abs(change).mean()),
        net_change=float(change.mean()),
        correlation_with_initial=float(np.corrcoef(flat, start)[0, 1]),
        histogram=np.histogram(flat, bins=bins)[0].tolist(),
        afferent_sum=weights.sum(axis=0),
    )


def presentation_steps(step, base, steps):
    """Trajectory-global spike steps for one presentation window.

    Continued simulations report global steps; window-relative steps are
    shifted. Every spike must then fall inside [base, base + steps).
    """
    step = np.asarray(step, dtype=np.int64)
    if len(step) and step.min() < base:
        step = step + base
    if len(step) and (step.min() < base or step.max() >= base + steps):
        raise ValueError("Spike steps fall outside their presentation")
    return step


class PlasticRecord:
    """Weight evidence after every presentation of one plastic trajectory."""

    def __init__(self, cfg, params, initial, tracked, evaluate, tag, scaled, length):
        self.initial, self.tracked, self.evaluate, self.tag = (
            initial,
            tracked,
            evaluate,
            tag,
        )
        self.scaled = scaled
        self.bins = np.linspace(0, params["w_max_us"], cfg["weight_histogram_bins"] + 1)
        self.keep = set(recipe.snapshot_indices(length, cfg["weight_snapshot_count"]))
        self.points = set(recipe.accuracy_boundaries(length, cfg["accuracy_every"])) - {
            0
        }
        self.summaries = [weight_summary(initial, initial, self.bins)]
        self.matrices = {0: initial.copy()}
        self.trajectory = [initial.ravel()[tracked]]
        self.scaling, self.curve = [], {}

    def update(self, boundary, current, state):
        """Record the matrix after `boundary` presentations; evaluate when due."""
        self.summaries.append(weight_summary(current, self.initial, self.bins))
        self.trajectory.append(current.ravel()[self.tracked])
        if boundary in self.keep:
            self.matrices[boundary] = current.copy()
        if self.scaled:
            key = f"plasticity/{recipe.PLASTIC_PROJECTION}/trace/scaling_rate"
            self.scaling.append(state.custom_state[key].cpu().numpy().ravel())
        if self.evaluate is not None and boundary in self.points:
            self.curve[boundary] = self.evaluate(current, f"{self.tag}-{boundary:04d}")

    def save(self, out, state, presentations):
        s = self.summaries
        np.savez_compressed(
            out / "weights.npz",
            boundaries=np.arange(presentations + 1),
            mean=[x["mean"] for x in s],
            std=[x["std"] for x in s],
            skew=[x["skew"] for x in s],
            quantiles=[x["quantiles"] for x in s],
            mean_abs_change=[x["mean_abs_change"] for x in s],
            net_change=[x["net_change"] for x in s],
            correlation_with_initial=[x["correlation_with_initial"] for x in s],
            histogram=[x["histogram"] for x in s],
            histogram_edges=self.bins,
            afferent_sum=np.stack([x["afferent_sum"] for x in s]),
            scaling_rate_hz=np.stack(self.scaling)
            if self.scaling
            else np.empty((0, 0)),
            tracked_index=self.tracked,
            tracked_weights=np.stack(self.trajectory),
        )
        np.savez_compressed(
            out / "weight_snapshots.npz",
            boundaries=np.array(sorted(self.matrices)),
            **{
                f"w{k:04d}": self.matrices[k].astype(np.float32)
                for k in sorted(self.matrices)
            },
        )
        sim.save_runtime_state(out / "final_runtime_state.pt", state)


def run_condition(
    cfg,
    run,
    condition,
    bundle,
    checkpoint,
    pixels,
    plan,
    hashes,
    *,
    request=None,
    record=None,
):
    """One continuous trajectory over a stream; returns E/I spike counts."""
    performance = sim.Performance(**PERFORMANCE)
    out = run.export / condition["id"]
    out.mkdir()
    pops = recipe.populations(cfg["networks"][condition["network"]]["architecture"])
    steps, total = cfg["steps"], len(plan)
    times = {p: [] for p in pops}
    cells = {p: [] for p in pops}
    counts = {p: np.zeros(total, dtype=np.int64) for p in pops}
    scores = np.zeros((total, cfg["n_output"]), dtype=np.float32)
    seen = hashes.setdefault(condition["stream"], [])
    state = None
    for index, item in enumerate(plan):
        drive = encode(cfg, pixels, item, index)
        digest = hashlib.sha256(drive.numpy().tobytes()).hexdigest()
        if len(seen) <= index:
            seen.append(digest)
        elif seen[index] != digest:
            raise ValueError("Stream input differs between conditions")
        events = {p: [] for p in pops}

        def collect(block, events=events):
            if block.kind == "spike_events":
                events[block.signal.split(".")[0]].append(block.values.cpu().numpy())

        result = sim.simulate(
            bundle=bundle,
            checkpoint=checkpoint,
            input_bindings=[sim.DenseArrayBinding("drive", drive)],
            seed=cfg["seed"],
            device=cfg["device"],
            performance=performance,
            plasticity=request,
            runtime_state=state,
            snapshot=sim.SnapshotSpec(out / "start", stage="start", parameters=True)
            if index == 0
            else None,
            diagnostics=False,
            recording=sim.RecordingSpec(
                signals=tuple(
                    sim.SignalRecording(p + ".spikes", kind="spike_events")
                    for p in pops
                ),
                sink=collect,
                sink_id=f"{recipe.SLUG}.stream/v1",
                block_steps=steps,
            ),
            retain_parameters=request is not None,
            retain_final_state=False,
            retain_runtime_state=True,
        )
        state = result.runtime_state
        base = index * steps
        if state.completed_steps != base + steps:
            raise ValueError("Continued trajectory lost its step count")
        for p in pops:
            rows = np.concatenate(events[p]) if events[p] else np.empty((0, 3))
            measured = int(result.outputs[p + "_count"].sum().item())
            if len(rows) != measured:
                raise ValueError("Raster and population counts differ")
            counts[p][index] = measured
            step = presentation_steps(rows[:, 0], base, steps)
            times[p].append((step * cfg["dt_ms"]).astype(np.float32))
            cells[p].append(rows[:, 2].astype(np.int16))
        scores[index] = result.outputs["class_scores"].cpu().numpy().reshape(-1)
        if record is not None:
            weight = result.parameters[recipe.PLASTIC_PROJECTION + ".weight"]
            record.update(index + 1, weight.numpy(), state)
        if (index + 1) % 10 == 0 or index + 1 == total:
            rate = counts["E"][index] / cfg["n_e"] / (cfg["presentation_ms"] / 1000)
            print(
                f"  {condition['id']}: {index + 1}/{total} presentations, E {rate:.2f} Hz",
                flush=True,
            )
    np.savez_compressed(
        out / "spikes.npz",
        **{p + "_time_ms": np.concatenate(times[p]) for p in pops},
        **{p + "_cell": np.concatenate(cells[p]) for p in pops},
        **{p + "_counts": counts[p] for p in pops},
        stream_scores=scores,
        stream_labels=np.array([item["label"] for item in plan], dtype=np.int64),
        stream_images=np.array([item["image"] for item in plan], dtype=np.int64),
        stream_rates_hz=np.array([item["rate_hz"] for item in plan], dtype=np.float64),
    )
    if record is not None:
        record.save(out, state, total)
    return counts


def accuracy(cfg, bundle, parameters, images, labels, scratch, tag):
    """Frozen evaluation on the fixed official-test subset (exp173 protocol)."""
    performance = sim.Performance(**PERFORMANCE)
    path = scratch / f"{tag}.pth"
    write_checkpoint(bundle, parameters, path, performance)
    generator = torch.Generator().manual_seed(cfg["accuracy_encoder_seed"])
    predictions, hashes = [], []
    for start in range(0, len(labels), cfg["accuracy_batch_size"]):
        batch = images[start : start + cfg["accuracy_batch_size"]]
        drive = sim.encode_images_poisson(
            torch.from_numpy(batch),
            cfg["steps"],
            cfg["dt_ms"],
            cfg["input_rate_hz"],
            generator=generator,
        )
        hashes.append(hashlib.sha256(drive.numpy().tobytes()).hexdigest())
        with torch.inference_mode():
            result = sim.simulate(
                bundle=bundle,
                checkpoint=path,
                input_bindings=[sim.DenseArrayBinding("drive", drive)],
                seed=cfg["seed"],
                device=cfg["device"],
                performance=performance,
                diagnostics=False,
                retain_parameters=False,
                retain_final_state=False,
                retain_runtime_state=False,
            )
        scores = result.outputs["class_scores"].cpu().numpy()
        if (
            scores.shape != (len(batch), cfg["n_output"])
            or not np.isfinite(scores).all()
        ):
            raise ValueError("Invalid classification scores")
        predictions.append(scores.argmax(axis=1))
    predictions = np.concatenate(predictions) if predictions else np.empty(0, int)
    return predictions, hashes


def save_accuracy(out, curve, labels):
    """Predictions of the fixed test images at each evaluated boundary."""
    boundaries = np.array(sorted(curve), dtype=np.int64)
    predictions = np.stack([curve[b] for b in boundaries])
    np.savez_compressed(
        out / "accuracy.npz",
        boundaries=boundaries,
        predictions=predictions,
        correct=predictions == labels[None, :],
    )


def load_images(cfg):
    """Official MNIST test set: the accuracy subset and the disjoint stream pool."""
    _, pixels, _, labels = sim.load_dataset(
        "mnist", split=True, evaluation_only=True, evaluation_split="test"
    )
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    sample_ids = np.empty(0, dtype=np.int64)
    if cfg["accuracy_images"]:
        sample_ids = recipe.test_sample_indices(
            total_images=len(labels),
            samples=cfg["accuracy_images"],
            seed=cfg["accuracy_sample_seed"],
        )
    pool = recipe.stream_pool(len(labels), cfg["stream_seed"])
    if np.intersect1d(pool, sample_ids).size:
        raise ValueError("Stream images overlap the accuracy subset")
    cfg["test_set_sha256"] = hashlib.sha256(
        pixels.tobytes() + labels.tobytes()
    ).hexdigest()
    return pixels, labels, sample_ids, pool


def stream_network(cfg, run, network, weights, data, hashes):
    """All conditions of one network: frozen first for each stream, then plastic."""
    pixels, labels, sample_ids, pool = data
    test_pixels, test_labels = pixels[sample_ids], labels[sample_ids]
    spec = cfg["networks"][network]
    performance = sim.Performance(**PERFORMANCE)
    bundle = recipe.author_network(cfg, network)
    (run.export / network).mkdir()
    bundle.write(run.export / network / "network.bundle", visualise=False)
    bundle.visualise(run.export / f"circuit-{network}.svg", view="circuit")
    parameters = recipe.native_weights(weights, spec["architecture"])
    checkpoint = run.scratch / f"{network}.pth"
    write_checkpoint(bundle, parameters, checkpoint, performance)
    key = recipe.PLASTIC_PROJECTION + ".weight"
    initial = parameters[key].numpy().copy()

    def evaluate(matrix, tag):
        if not len(test_labels):
            return np.empty(0, dtype=np.int64)
        evaluated = dict(parameters)
        evaluated[key] = torch.from_numpy(np.ascontiguousarray(matrix, np.float32))
        predictions, batch_hashes = accuracy(
            cfg, bundle, evaluated, test_pixels, test_labels, run.scratch, tag
        )
        if not hashes["accuracy"]:
            hashes["accuracy"].extend(batch_hashes)
        elif batch_hashes != hashes["accuracy"]:
            raise ValueError("Accuracy inputs differ between evaluations")
        share = 100 * (predictions == test_labels).mean()
        print(
            f"  {tag}: test accuracy {share:.1f}% on {len(test_labels)} images",
            flush=True,
        )
        return predictions

    trained = evaluate(initial, f"{network}-trained")
    mine = [c for c in recipe.conditions(cfg) if c["network"] == network]
    rules, targets, timings = {}, {}, {}
    for kind in dict.fromkeys(c["stream"] for c in mine):
        plan = recipe.stream_plan(cfg, kind, pool, labels, cfg["input_rate_hz"])
        group = sorted(
            (c for c in mine if c["stream"] == kind),
            key=lambda c: c["rule"] != "frozen",
        )
        if group[0]["rule"] != "frozen":
            raise ValueError(f"{network}/{kind}: every stream needs a frozen condition")
        for condition in group:
            tick = time.perf_counter()
            print(f"{condition['id']}: streaming {len(plan)} presentations", flush=True)
            params = recipe.rule_parameters(cfg, initial, condition["tau_stdp_ms"])
            target = targets.get(kind, 0.0) * condition["setpoint"]
            request = recipe.plasticity_request(cfg, condition["rule"], params, target)
            record = None
            if request is not None:
                tracked = recipe.tracked_synapses(
                    initial.shape, cfg["tracked_synapses"], cfg["tracked_synapse_seed"]
                )
                record = PlasticRecord(
                    cfg,
                    params,
                    initial,
                    tracked,
                    evaluate,
                    condition["id"],
                    scaled=condition["rule"] in ("scaling", "both"),
                    length=len(plan),
                )
            counts = run_condition(
                cfg,
                run,
                condition,
                bundle,
                checkpoint,
                pixels,
                plan,
                hashes["stream"],
                request=request,
                record=record,
            )
            out = run.export / condition["id"]
            if request is not None:
                write_json_atomic(out / "plasticity.json", request.json())
            if condition["rule"] == "frozen":
                start, stop = recipe.setpoint_window(cfg, kind)
                seconds = (stop - start) * cfg["presentation_ms"] / 1000
                targets[kind] = float(
                    counts["E"][start:stop].sum() / cfg["n_e"] / seconds
                )
            # Boundary 0 is the trained matrix: identical weights and test inputs.
            save_accuracy(
                out, {0: trained, **(record.curve if record else {})}, test_labels
            )
            rules[condition["id"]] = dict(params, target_rate_hz=target or None)
            timings[condition["id"]] = time.perf_counter() - tick
            print(
                f"{condition['id']}: done in {timings[condition['id']]:.1f}s",
                flush=True,
            )
    return rules, targets, timings


def execute(cfg, *, run_id=None):
    started = time.perf_counter()
    if cfg["threads"]:
        torch.set_num_threads(cfg["threads"])
    cfg["threads_used"] = torch.get_num_threads()
    source, weights = load_source(cfg)
    data = load_images(cfg)
    cfg["conditions"] = recipe.conditions(cfg)
    cfg["accuracy_boundaries"] = recipe.accuracy_boundaries(
        cfg["presentations"], cfg["accuracy_every"]
    )
    with stage_run(
        ROOT,
        recipe.SLUG,
        "compute",
        inputs={"bank": source},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        run.details["scientific_artifacts"] = {
            "snapshot_stages": {c["id"]: ["start"] for c in cfg["conditions"]}
        }
        pixels, labels, sample_ids, pool = data
        np.savez_compressed(
            run.export / "dataset.npz",
            accuracy_labels=labels[sample_ids],
            test_sample_indices=sample_ids,
            stream_pool=pool,
        )
        rules, targets, timings = {}, {}, {}
        hashes = {"stream": {}, "accuracy": []}
        for network in cfg["source_cells"]:
            spent_rules, spent_targets, spent = stream_network(
                cfg, run, network, weights[network], data, hashes
            )
            rules.update(spent_rules)
            targets[network] = spent_targets
            timings.update(spent)
        write_json_atomic(
            run.export / "rules.json", dict(rules=rules, scaling_targets_hz=targets)
        )
        write_json_atomic(
            run.export / "input-hashes.json",
            dict(
                stream_sha256=hashes["stream"], accuracy_batch_sha256=hashes["accuracy"]
            ),
        )
        run.details["execution"].update(
            elapsed_s=time.perf_counter() - started, condition_seconds=timings
        )
    print(run.run_id, flush=True)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentations", type=int, help="stream length (default 50)")
    parser.add_argument(
        "--accuracy-images", type=int, help="test images (default 500; 0 skips)"
    )
    parser.add_argument(
        "--accuracy-every", type=int, help="test every N presentations (default 10)"
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    overrides = {
        key: value
        for key, value in dict(
            presentations=args.presentations,
            accuracy_images=args.accuracy_images,
            accuracy_every=args.accuracy_every,
        ).items()
        if value is not None
    }
    execute(recipe.configuration(**overrides), run_id=args.run_id)


if __name__ == "__main__":
    main()
