"""Scoped compatibility adapters for native, differentiable masked spike counts.

SNNLab 0.32.3's dataset loader rejects 2D masks, and its executor enables online
reductions only without autograd. Native sum accumulation itself preserves
gradients. Limit this adapter to exp175's single masked sum; select-final uses
detachment and must not be enabled for training through this adapter.
"""

from contextlib import contextmanager
from unittest.mock import patch

from snnlab.sim.graph import forward_pass
from snnlab.sim.inputs import resolution
from snnlab.sim.runtime.recording import OnlineReductions


def differentiable_counts(graph, axes, parameters, **options):
    if graph.get("name") == "mnist_fixed_contrast_finetuning":
        reductions = [
            op
            for op in graph["operations"]
            if op["kind"].startswith("reduce_") or op["kind"] == "select_final"
        ]
        if len(reductions) != 1 or (reductions[0]["id"], reductions[0]["kind"]) != (
            "readout_counts",
            "reduce_sum",
        ):
            raise ValueError(
                "Streaming adapter requires exactly exp175's masked spike sum"
            )
        options["enabled"] = True
    return OnlineReductions(graph, axes, parameters, **options)


@contextmanager
def native_spike_counts():
    original = resolution._dataset_batch_values

    def dataset_values(spikes, *, binding, cap):
        # Add/remove an axis only around the loader's 3D-only validation. The
        # native graph and reducer receive the original (time, batch) mask.
        if (
            isinstance(spikes, dict)
            and "score_window" in spikes
            and spikes["score_window"].ndim == 2
        ):
            values = dict(spikes, score_window=spikes["score_window"].unsqueeze(-1))
            tensors, state = original(values, binding=binding, cap=cap)
            tensors["score_window"] = tensors["score_window"].squeeze(-1)
            return tensors, state
        return original(spikes, binding=binding, cap=cap)

    with (
        patch.object(resolution, "_dataset_batch_values", dataset_values),
        patch.object(forward_pass, "OnlineReductions", differentiable_counts),
    ):
        yield
