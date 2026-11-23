"""Load unpenalised TR02 PING weights and fine-tune the encoder/decoder."""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("PINGLAB_DATASET_CACHE", ROOT / ".pingstore/datasets"))
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]

import numpy as np
import torch
from experiments.exp175 import dataset, recipe
from experiments.exp175.custom.snapshots import capture_selected
from experiments.exp175.custom.streaming import native_spike_counts
from experiments.exp175.custom.transfer import inference_checkpoint
from pingstore.library import Store, source_run, stage_run, utc_now, write_json_atomic
from snnlab import sim


def load_tr02(cfg):
    source = source_run(
        ROOT / ".pingstore", cfg["source_run"], experiment=recipe.SLUG, stage="import"
    )
    original = json.loads(source.file("config.json").read_text())
    if (
        original["training_run_id"],
        original["model"],
        original["fr_reg_upper_strength"],
    ) != ("TR-02", "ping", 0.0):
        raise ValueError("Expected unpenalised TR02 PING")
    cfg.update(
        tau_ampa_ms=original["tau_ampa_ms"],
        tau_gaba_ms=original["tau_gaba_ms"],
        surrogate_slope=original["surrogate_slope"],
        voltage_grad_dampen=original["v_grad_dampen"],
        target_image_file=str(DATASETS / "MNIST/raw/train-images-idx3-ubyte"),
    )
    raw = torch.load(
        source.file("weights.pth"),
        map_location="cpu",
        weights_only=True,
    )
    return source, recipe.initialise(raw, cfg)


def prepare_tr02_input(cfg):
    """Export only the three used TR02 matrices and required source settings."""
    store = Store(ROOT / ".pingstore")
    if (store.runs / cfg["source_run"]).exists():
        source_run(
            store.root, cfg["source_run"], experiment=recipe.SLUG, stage="import"
        )
        print(cfg["source_run"])
        return
    source = source_run(
        store.root, cfg["tr02_run"], experiment="exp022", stage="compute"
    )
    original = json.loads(source.file(cfg["source_cell"], "config.json").read_text())
    raw = torch.load(
        source.file(cfg["source_cell"], "weights.pth"),
        map_location="cpu",
        weights_only=True,
    )
    if (
        original["training_run_id"],
        original["model"],
        original["fr_reg_upper_strength"],
    ) != ("TR-02", "ping", 0.0):
        raise ValueError("Expected unpenalised TR02 PING")
    if any(raw[key].count_nonzero() for key in ("W_ee.1", "W_ii.1")):
        raise ValueError("The omitted frozen recurrent matrices must be zero")
    required = {key: raw[key] for key in ("W_ff.0", "W_ei.1", "W_ie.1")}
    settings = {
        key: original[key]
        for key in (
            "training_run_id",
            "model",
            "fr_reg_upper_strength",
            "tau_ampa_ms",
            "tau_gaba_ms",
            "surrogate_slope",
            "v_grad_dampen",
        )
    }
    settings.update(source_run=source.run_id, source_cell=cfg["source_cell"])
    with tempfile.TemporaryDirectory(prefix="exp175-tr02-") as temporary:
        export = Path(temporary) / "export"
        export.mkdir()
        torch.save(required, export / "weights.pth")
        write_json_atomic(export / "config.json", settings)
        store.insert(
            temporary,
            record=dict(
                id=cfg["source_run"],
                experiment=recipe.SLUG,
                stage="import",
                created_at=utc_now(),
                description="Only the unpenalised TR02 encoder, E/I feedback weights and required settings",
                depends_on=[source.run_id],
            ),
        )
    print(cfg["source_run"])


def verify_parameters(final, initial, *, require_change):
    if set(final) != set(initial):
        raise ValueError("Final parameter membership changed")
    changes = {}
    for key, value in final.items():
        current = torch.as_tensor(value).detach().cpu()
        if not torch.isfinite(current).all() or (current < 0).any():
            raise ValueError(f"Invalid trained parameter: {key}")
        equal = torch.equal(current, initial[key])
        if key in recipe.FROZEN and not equal:
            raise ValueError(f"Frozen attention weight changed: {key}")
        if require_change and key in recipe.TRAINABLE and equal:
            raise ValueError(f"Trainable weight did not change: {key}")
        changes[key] = float((current - initial[key]).square().mean().sqrt())
    return changes


def record_examples(run, bundle, path, arrays, cfg):
    """After future training, retain the first two validation targets in all cue conditions."""
    rows = list(range(cfg["train_images"], cfg["train_images"] + 8))
    binding = replace(
        dataset.binding(path, cfg, "validation"), sample_indices=tuple(rows)
    )
    with native_spike_counts():
        result = sim.simulate(
            bundle=bundle,
            checkpoint=run.export / "best.checkpoint",
            input_bindings=[binding],
            device=cfg["device"],
            batch_size=1,
            seed=cfg["seed"],
            diagnostics=False,
            performance=sim.Performance(
                compile_mode=cfg["compile_mode"], precision=cfg["precision"]
            ),
            snapshot=sim.SnapshotSpec(
                run.export / "examples-start", stage="start", parameters=True
            ),
            retain_parameters=False,
            retain_final_state=False,
            retain_runtime_state=False,
        )
    a, b, target, cue, phase = [], [], [], [], []
    for row in rows:
        pair, sender, window = dataset.presentation(arrays, row, cfg)
        a.append(arrays["features"][pair[0]])
        b.append(arrays["features"][pair[1]])
        target.append(arrays["features"][row])
        cue.append(sender)
        phase.append(window)
    np.savez_compressed(
        run.export / "reconstructions.npz",
        images_A=np.asarray(a) / 255.0,
        images_B=np.asarray(b) / 255.0,
        targets=np.asarray(target) / 255.0,
        cue=np.asarray(cue),
        phase=np.asarray(phase),
        reconstructed=result.outputs["pixels"].detach().cpu().numpy(),
        counts=result.outputs["counts"].detach().cpu().numpy(),
    )


def compute(cfg, *, run_id=None):
    started = time.perf_counter()
    if torch.device(cfg["device"]).type == "mps":
        torch.mps.set_per_process_memory_fraction(cfg["mps_memory_fraction"])
    torch.set_num_threads(cfg["threads"])
    source, parameters = load_tr02(cfg)
    pixels, labels, pins = dataset.read_training(DATASETS / "MNIST/raw")
    arrays = dataset.snapshot_arrays(pixels, labels, cfg)
    cfg["dataset_sha256"] = pins
    with stage_run(
        ROOT,
        recipe.SLUG,
        "compute",
        inputs={"tr02": source},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        print(f"Run: {run.directory}", flush=True)
        bundle = recipe.author_network(cfg)
        bundle.write(run.export / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        path = run.export / "dataset.npz"
        np.savez_compressed(path, **arrays)
        with native_spike_counts():
            result = sim.train(
                bundle=bundle,
                weights=sim.WeightInitialization(parameters),
                input_bindings=[dataset.binding(path, cfg, "train")],
                validation=sim.ValidationSpec(
                    input_bindings=[dataset.binding(path, cfg, "validation")],
                    encoding_seeds=tuple(range(cfg["validation_draws"])),
                ),
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                batching=sim.Batching(
                    order="shuffle", microbatch_size=cfg["microbatch_size"]
                ),
                seed=cfg["seed"],
                device=cfg["device"],
                performance=sim.Performance(
                    compile_mode=cfg["compile_mode"], precision=cfg["precision"]
                ),
                training_timing=True,
                diagnostics=False,
                metrics=sim.TrainingMetrics(train="online"),
                checkpoint_selection=sim.CheckpointSelection(
                    metric=sim.SelectionMetric("loss", direction="min"),
                    include_initial=True,
                ),
                save_final_checkpoint=run.export / "final.checkpoint",
                save_selected_checkpoint=run.export / "best.checkpoint",
                snapshot=sim.SnapshotSpec(
                    run.export / "final-training", stage="end", parameters=True
                ),
                retain_final_state=False,
                retain_runtime_state=False,
                on_epoch_end=lambda row: print(
                    f"epoch {row.epoch}, updates {row.updates}", flush=True
                ),
            )
        final = sim.load_training_checkpoint(run.export / "final.checkpoint")
        selected = sim.load_training_checkpoint(run.export / "best.checkpoint")
        changes = verify_parameters(final.parameters, parameters, require_change=True)
        verify_parameters(selected.parameters, parameters, require_change=False)
        counts = result.metrics["batch_counts"]
        if counts["optimizer_updates"] != cfg["epochs"] * math.ceil(
            cfg["train_images"] / cfg["batch_size"]
        ):
            raise ValueError("Incomplete optimiser schedule")
        if counts["examples"] != cfg["epochs"] * cfg["train_images"]:
            raise ValueError("Incomplete unique-target presentation schedule")
        capture_selected(bundle, selected, cfg, run.export / "best-validation")
        record_examples(run, bundle, path, arrays, cfg)
        if cfg["architecture"] == "single":
            evaluate_transfer(run, bundle, selected, path, arrays, cfg)
        write_json_atomic(
            run.export / "metrics.json",
            {
                key: result.metrics[key]
                for key in (
                    "epochs",
                    "batch_counts",
                    "training_timing",
                    "performance",
                    "selection_record",
                )
            },
        )
        write_json_atomic(
            run.export / "checks.json",
            dict(
                final_weight_rms_changes=changes,
                all_frozen_weights_identical=True,
                shared_encoder=True,
                official_test_used=False,
                selected_epoch=int(selected.data_state["epoch"]),
                limitation="TR02 already used this validation split for source selection; not an independent test",
            ),
        )
        write_json_atomic(
            run.export / "timing.json",
            dict(total_seconds=time.perf_counter() - started),
        )


def transfer_parameters(base, selected):
    """Preserve the relay body and copy only the standalone encoder/decoder."""
    result = {name: value.clone() for name, value in base.items()}
    for name in recipe.TRAINABLE:
        value = selected[name].detach().cpu()
        if value.shape != result[name].shape:
            raise ValueError(f"Transfer shape mismatch: {name}")
        result[name] = value.clone()
    for sender in "ABC":
        for role in recipe.LOCAL_ROLES:
            if not torch.equal(
                result[f"{sender}_{role}.weight"], selected[f"A_{role}.weight"].cpu()
            ):
                raise ValueError("Standalone and relay PING bodies differ")
    return result


def evaluate_transfer(run, single_bundle, selected, path, arrays, cfg):
    """Paired single-draw validation, without any relay training or selection."""
    started = time.perf_counter()
    relay_cfg = dict(cfg, architecture="relay")
    _, base = load_tr02(relay_cfg)
    transferred = transfer_parameters(base, selected.parameters)
    relay_bundle = recipe.author_network(relay_cfg)
    relay_bundle.write(run.export / "relay.bundle", visualise=False)
    relay_bundle.visualise(run.export / "relay-circuit.svg", view="circuit")
    inference_checkpoint(
        relay_bundle,
        transferred,
        run.export / "relay-transfer.checkpoint",
        seed=cfg["seed"],
        source_epoch=int(selected.data_state["epoch"]),
    )
    rows = np.arange(cfg["train_images"], len(arrays["ids"]))
    expected = arrays["features"][rows] / 255.0
    comparison = {}
    outputs = {}
    for name, bundle, configuration, checkpoint in (
        ("single", single_bundle, cfg, run.export / "best.checkpoint"),
        ("relay", relay_bundle, relay_cfg, run.export / "relay-transfer.checkpoint"),
    ):
        with native_spike_counts():
            result = sim.simulate(
                bundle=bundle,
                checkpoint=checkpoint,
                input_bindings=[dataset.binding(path, configuration, "validation")],
                device=cfg["device"],
                batch_size=cfg["batch_size"],
                seed=cfg["seed"],
                diagnostics=False,
                performance=sim.Performance(
                    compile_mode=cfg["compile_mode"], precision=cfg["precision"]
                ),
                snapshot=sim.SnapshotSpec(
                    run.export / f"{name}-evaluation-start",
                    stage="start",
                    parameters=True,
                ),
                retain_parameters=False,
                retain_final_state=False,
                retain_runtime_state=False,
            )
        prediction = result.outputs["pixels"].detach().cpu().numpy()
        losses = np.mean((prediction - expected) ** 2, axis=1)
        outputs[name] = prediction
        comparison[name] = dict(
            pixel_mse=float(losses.mean()),
            per_digit_mse={
                str(digit): float(losses[arrays["digits"][rows] == digit].mean())
                for digit in range(10)
            },
        )
        print(f"{name} paired validation MSE: {losses.mean():.6f}", flush=True)
    images_a, images_b, cues, phases = [], [], [], []
    for row in rows:
        pair, sender, phase = dataset.presentation(arrays, int(row), relay_cfg)
        images_a.append(arrays["features"][pair[0]] / 255.0)
        images_b.append(arrays["features"][pair[1]] / 255.0)
        cues.append(sender)
        phases.append(phase)
    np.savez_compressed(
        run.export / "transfer-reconstructions.npz",
        rows=rows,
        original_ids=arrays["original_ids"][rows],
        digits=arrays["digits"][rows],
        images_A=np.asarray(images_a),
        images_B=np.asarray(images_b),
        targets=expected,
        cue=np.asarray(cues),
        phase=np.asarray(phases),
        single=outputs["single"],
        relay=outputs["relay"],
    )
    write_json_atomic(
        run.export / "transfer-metrics.json",
        dict(
            comparison=comparison,
            validation_images=cfg["validation_images"],
            presentations=len(rows),
            encoding_draws=1,
            protocol="Paired simulate-phase draw, distinct from three-draw training validation",
            checkpoint_epoch=int(selected.data_state["epoch"]),
            relay_finetuning_updates=0,
            transferred_parameters=list(recipe.TRAINABLE),
            all_relay_frozen_weights_identical=all(
                torch.equal(transferred[k], base[k]) for k in recipe.FROZEN
            ),
            total_seconds=time.perf_counter() - started,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Reserved Pingstore compute identity")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "plan", help="Print the resolved protocol; no files or simulation"
    )
    commands.add_parser(
        "prepare-input", help="Create the small TR02 input export; no simulation"
    )
    trainer = commands.add_parser(
        "train", help="Explicitly execute training; requires separate authorisation"
    )
    defaults = recipe.configuration()
    trainer.add_argument(
        "--device", default=os.environ.get("SNNLAB_DEVICE", defaults["device"])
    )
    trainer.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("PINGLAB_CPU_THREADS", defaults["threads"])),
    )
    trainer.add_argument("--architecture", choices=("relay", "single"), default="relay")
    trainer.add_argument("--epochs", type=int, default=defaults["epochs"])
    trainer.add_argument("--train-images", type=int, default=defaults["train_images"])
    trainer.add_argument(
        "--validation-images", type=int, default=defaults["validation_images"]
    )
    trainer.add_argument(
        "--duration-ms",
        type=float,
        help="Short pilot: switch halfway; score at 25–45%% and 75–95%%",
    )
    trainer.add_argument(
        "--microbatch-size", type=int, default=defaults["microbatch_size"]
    )
    trainer.add_argument(
        "--compile-mode", choices=("none", "default"), default=defaults["compile_mode"]
    )
    args = parser.parse_args()
    if args.command == "plan":
        print(
            json.dumps(
                dict(
                    configuration=recipe.configuration(),
                    trainable=recipe.TRAINABLE,
                    frozen=recipe.FROZEN,
                    status="Scaffold; not executed",
                ),
                indent=2,
            )
        )
    elif args.command == "prepare-input":
        prepare_tr02_input(recipe.configuration())
    else:
        timing = (
            {}
            if args.duration_ms is None
            else dict(
                duration_ms=args.duration_ms,
                switch_ms=args.duration_ms / 2,
                windows_ms=(
                    (args.duration_ms * 0.25, args.duration_ms * 0.45),
                    (args.duration_ms * 0.75, args.duration_ms * 0.95),
                ),
            )
        )
        compute(
            recipe.configuration(
                architecture=args.architecture,
                device=args.device,
                threads=args.threads,
                epochs=args.epochs,
                train_images=args.train_images,
                validation_images=args.validation_images,
                microbatch_size=args.microbatch_size,
                compile_mode=None if args.compile_mode == "none" else args.compile_mode,
                **timing,
            ),
            run_id=args.run_id,
        )


if __name__ == "__main__":
    main()
