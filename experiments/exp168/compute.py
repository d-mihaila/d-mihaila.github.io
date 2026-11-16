"""Simulate the EE/II-on comparison, or run a short explicit diagnostic."""

import argparse
import shutil
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp168 import dataset, recipe
from pingstore.library import load_json, source_run, stage_run


def execute(cfg, directory, diagram, *, bundle=None, checkpoint=None):
    from snnlab.sim import (
        DenseArrayBinding,
        Performance,
        RecordingSpec,
        SignalRecording,
        SnapshotSpec,
        simulate,
    )

    directory.mkdir(parents=True)
    bundle = bundle if bundle is not None else recipe.author_network(cfg)
    bundle.write(directory / "network.bundle", visualise=False)
    if diagram:
        bundle.visualise(directory.parent / "circuit.svg", view="circuit")
    drive = dataset.encode_drive(cfg).to(cfg["device"])
    events = {"E": [], "I": []}

    def collect(block):
        if block.kind == "spike_events":
            events[block.signal.split(".")[0]].append(block.values.numpy())

    simulate(
        bundle=bundle,
        input_bindings=(DenseArrayBinding("X", drive),),
        seed=cfg["seed"],
        checkpoint=checkpoint,
        device=cfg["device"],
        performance=Performance(**cfg["performance"]),
        diagnostics=False,
        recording=RecordingSpec(
            signals=tuple(
                SignalRecording(p + ".spikes", kind="spike_events") for p in ("E", "I")
            ),
            sink=collect,
            sink_id="exp168.sparse-spikes/v2",
            block_steps=cfg["recording_block_steps"],
        ),
        snapshot=SnapshotSpec(directory / "start", stage="start", parameters=True),
        retain_outputs=False,
        retain_parameters=True,
        retain_final_state=False,
        retain_runtime_state=False,
    )
    arrays = {}
    for population, blocks in events.items():
        rows = np.concatenate(blocks) if blocks else np.empty((0, 3))
        arrays[population + "_time_ms"] = rows[:, 0] * cfg["dt_ms"]
        arrays[population + "_cell"] = rows[:, 2]
    np.savez_compressed(directory / "events.npz", **arrays)


def import_transition(directory, *, source=recipe.SOURCES["on"]["run_id"], run_id=None):
    """Validate and retain completed evidence, without executing a simulation."""
    import torch
    from snnlab.sim import GraphExecutor, plan_graph

    started = perf_counter()
    directory = Path(directory).resolve()
    origin = source_run(
        REPO / ".pingstore", source, experiment=recipe.SLUG, stage="compute"
    )
    cfg = recipe.transition_configuration(
        recipe.case_configuration(origin, "on")["PING"]
    )
    graph = load_json(
        recipe.case_file(origin, "on", "PING", "network.bundle", "graph.json")
    )
    recorded_graph = load_json(directory / "network.bundle" / "graph.json")
    if recorded_graph != recipe.transition_graph(graph, cfg):
        raise ValueError("imported graph differs from the fixed-wiring gain protocol")
    summary = load_json(directory / "summary.json")
    if (
        summary["gain_E"],
        summary["gain_I"],
        summary["duration_ms"],
        summary["source_rate_hz"],
        summary["input_probability"],
    ) != (
        [1.0, 1.0],
        [1.0, 0.0],
        cfg["duration_ms"],
        cfg["external_rate_hz"],
        cfg["input_connection_probability"],
    ):
        raise ValueError("imported execution settings disagree with the protocol")
    snapshot = load_json(directory / "start" / "snapshot.json")
    if (
        snapshot["network"] != recorded_graph
        or snapshot["execution"]["duration_ms"] != cfg["duration_ms"]
    ):
        raise ValueError("native start snapshot disagrees with the imported execution")
    checkpoint = directory.parent / "fixed-weights.pt"
    torch.set_num_threads(cfg["torch_threads"])
    model = GraphExecutor(
        plan_graph(recorded_graph),
        seed=cfg["seed"],
        compile_mode=None,
        precision="highest",
        cuda_graphs="off",
    )
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    original_snapshot = load_json(
        recipe.case_file(origin, "on", "PING", "start", "snapshot.json")
    )
    with np.load(
        recipe.case_file(origin, "on", "PING", "start", "parameters.npz"),
        allow_pickle=False,
    ) as parameters:
        for name, weight in model.parameter_map().items():
            if not np.array_equal(
                weight.detach().numpy(),
                parameters[original_snapshot["parameters"][name]["key"]],
            ):
                raise ValueError(f"imported parameter differs from Figure 1: {name}")
    with np.load(directory / "events.npz", allow_pickle=False) as events:
        for population in ("E", "I"):
            times, cells = events[population + "_time_ms"], events[population + "_cell"]
            if (
                times.shape != cells.shape
                or not np.isfinite(times).all()
                or np.any(times < 0)
                or np.any(times >= cfg["duration_ms"])
                or np.any(cells < 0)
                or np.any(cells >= cfg["n_" + population.lower()])
            ):
                raise ValueError("imported spike events have invalid times or cells")
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        inputs={"fixed_network": origin},
        configuration={"transition": cfg},
        operation="import-completed-transition",
        run_id=run_id,
    ) as run:
        for name in ("network.bundle", "start"):
            shutil.copytree(directory / name, run.export / name)
        for name in ("events.npz", "summary.json"):
            shutil.copy2(directory / name, run.export / name)
        shutil.copy2(checkpoint, run.export / "fixed-weights.pt")
        # Historical snapshot bytes retain their original checkpoint path; replay
        # uses the retained run-local checkpoint, never that historical path.
        run.details["scientific_artifacts"] = dict(
            checkpoint="fixed-weights.pt",
            transition_prefix="",
            parameter_source=dict(run_id=origin.run_id, snapshot="PING/start"),
            validation=dict(
                all_sampled_parameters_identical=True,
                graph_only_input_gain_changes=True,
            ),
        )
        run.details["scientific_execution"] = dict(
            elapsed_seconds=summary["elapsed_s"], origin="local"
        )
        run.details["import_provenance"] = dict(
            original_directory=str(directory),
            original_summary="summary.json",
            native_snapshot_checkpoint_path_is_historical=True,
        )
        run.details["execution"]["elapsed_s"] = perf_counter() - started
    return run.run_id


def simulate_transition(source, *, run_id=None):
    """Explicitly replay the retained transition using the original sampled weights."""
    import torch
    from snnlab import lang

    started = perf_counter()
    origin = source_run(
        REPO / ".pingstore", source, experiment=recipe.SLUG, stage="compute"
    )
    cfg = origin.details["execution"]["configuration"]["transition"]
    graph = load_json(recipe.transition_file(origin, "network.bundle", "graph.json"))
    torch.set_num_threads(cfg["torch_threads"])
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        inputs={"transition_template": origin},
        configuration={"transition": cfg},
        run_id=run_id,
    ) as run:
        execute(
            cfg,
            run.export / "transition",
            True,
            bundle=lang.Bundle(graph, None, {}, []),
            checkpoint=origin.file("fixed-weights.pt"),
        )
        shutil.copy2(origin.file("fixed-weights.pt"), run.export / "fixed-weights.pt")
        run.details["scientific_artifacts"] = dict(
            transition_prefix="transition", checkpoint="fixed-weights.pt"
        )
        run.details["execution"]["elapsed_s"] = perf_counter() - started
    return run.run_id


def simulate_conditions(
    *,
    device="cpu",
    duration_ms=2050.0,
    seed=43,
    ai_rate_hz=None,
    ai_probability=None,
    ai_gain=None,
    ping_rate_hz=None,
    ping_probability=None,
    ping_gain=None,
    run_id=None,
):
    from importlib.metadata import version

    import torch

    if duration_ms != 2050.0:
        raise ValueError("retained compute requires the requested 2050 ms duration")
    started = perf_counter()
    configs = {
        state: recipe.configuration(
            state,
            recurrence="on",
            device=device,
            seed=seed,
            external_rate_hz=ai_rate_hz if state == "AI" else ping_rate_hz,
            input_connection_probability=(
                ai_probability if state == "AI" else ping_probability
            ),
            input_i_gain=ai_gain if state == "AI" else ping_gain,
        )
        for state in recipe.CONDITIONS
    }
    torch.set_num_threads(configs["AI"]["torch_threads"])
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        configuration={"recurrence": "on", "conditions": configs},
        run_id=run_id,
    ) as run:
        run.details["scientific_artifacts"] = {
            "snapshot_stages": {state: ["start"] for state in recipe.CONDITIONS}
        }
        for index, (state, cfg) in enumerate(configs.items()):
            execute(cfg, run.export / state, index == 0)
        run.details["execution"].update(
            elapsed_s=perf_counter() - started,
            snnlab_version=version("snnlab"),
        )
    return run.run_id


def diagnostic(args):
    import tempfile

    import torch

    torch.set_num_threads(4)
    started = perf_counter()
    cfg = recipe.configuration(
        args.condition,
        recurrence="on",
        device=args.device,
        seed=args.seed,
        external_rate_hz=args.rate_hz,
        input_connection_probability=args.probability,
        input_i_gain=args.gain,
    )
    cfg["duration_ms"] = args.duration_ms
    cfg["measurement_ms"] = [50.0, args.duration_ms]
    with tempfile.TemporaryDirectory() as temporary:
        execute(cfg, Path(temporary) / "run", False)
        with np.load(
            Path(temporary) / "run" / "events.npz", allow_pickle=False
        ) as events:
            summary = {}
            for population in ("E", "I"):
                times = events[population + "_time_ms"]
                mask = (times >= 50.0) & (times < args.duration_ms)
                summary[population + "_rate_hz"] = float(
                    mask.sum()
                    / cfg["n_" + population.lower()]
                    / ((args.duration_ms - 50.0) / 1000)
                )
                summary[population + "_events"] = int(mask.sum())
    print(
        {
            "condition": args.condition,
            "rate_hz": args.rate_hz,
            "probability": args.probability,
            "gain": args.gain,
            "summary": summary,
            "elapsed_s": perf_counter() - started,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    simulate_parser = commands.add_parser("simulate")
    simulate_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    simulate_parser.add_argument("--seed", type=int, default=43)
    simulate_parser.add_argument("--ai-rate-hz", type=float)
    simulate_parser.add_argument("--ai-probability", type=float)
    simulate_parser.add_argument("--ai-gain", type=float)
    simulate_parser.add_argument("--ping-rate-hz", type=float)
    simulate_parser.add_argument("--ping-probability", type=float)
    simulate_parser.add_argument("--ping-gain", type=float)
    simulate_parser.add_argument("--run-id")
    diagnostic_parser = commands.add_parser("diagnostic")
    diagnostic_parser.add_argument("condition", choices=recipe.CONDITIONS)
    diagnostic_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    diagnostic_parser.add_argument("--seed", type=int, default=43)
    diagnostic_parser.add_argument("--duration-ms", type=float, default=205.0)
    diagnostic_parser.add_argument("--rate-hz", type=float)
    diagnostic_parser.add_argument("--probability", type=float)
    diagnostic_parser.add_argument("--gain", type=float)
    import_parser = commands.add_parser("import-transition")
    import_parser.add_argument("directory", type=Path)
    import_parser.add_argument("--source", default=recipe.SOURCES["on"]["run_id"])
    import_parser.add_argument("--run-id")
    transition_parser = commands.add_parser("transition")
    transition_parser.add_argument("--source", required=True)
    transition_parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.command == "simulate":
        simulate_conditions(
            device=args.device,
            seed=args.seed,
            ai_rate_hz=args.ai_rate_hz,
            ai_probability=args.ai_probability,
            ai_gain=args.ai_gain,
            ping_rate_hz=args.ping_rate_hz,
            ping_probability=args.ping_probability,
            ping_gain=args.ping_gain,
            run_id=args.run_id,
        )
    elif args.command == "import-transition":
        import_transition(args.directory, source=args.source, run_id=args.run_id)
    elif args.command == "transition":
        simulate_transition(args.source, run_id=args.run_id)
    else:
        diagnostic(args)
