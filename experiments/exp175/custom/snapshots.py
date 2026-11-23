"""Native lifecycle adapter: capture selected parameters without an inference run."""

import torch
from snnlab import sim
from snnlab.sim.artifacts.snapshots import SnapshotSession
from snnlab.sim.requests.spec import ExecutionSpec


def capture_selected(bundle, checkpoint, cfg, path):
    built = sim.build(
        bundle=bundle,
        training=bundle.training,
        seed=cfg["seed"],
        device="cpu",
        performance=sim.Performance(compile_mode=None),
    )
    actual = built.model.parameter_map()
    if set(actual) != set(checkpoint.parameters):
        raise ValueError("Selected checkpoint parameter membership mismatch")
    with torch.no_grad():
        for name, value in actual.items():
            value.copy_(checkpoint.parameters[name])
    spec = ExecutionSpec(
        kind="train",
        graph=bundle.graph,
        seed=cfg["seed"],
        device="cpu",
        snapshot=sim.SnapshotSpec(path, stage="end", parameters=True),
        training=bundle.training,
    )
    session = SnapshotSession(
        spec,
        built.model,
        kind="train",
        training=bundle.training,
        protocol={"timing": {"duration_ms": cfg["duration_ms"]}},
        device_resolved="cpu",
    )
    session.capture(
        "end",
        checkpoint_role="best_validation",
        epoch=checkpoint.data_state["epoch"],
        completed_updates=checkpoint.completed_updates,
    )
