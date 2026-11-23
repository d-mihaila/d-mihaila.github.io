"""Write a native inference-only checkpoint for the new relay architecture."""

import torch
from snnlab import sim
from snnlab.sim.artifacts.integrity import _json_digest


def inference_checkpoint(bundle, parameters, path, *, seed, source_epoch):
    """Transfer weights explicitly; discard optimiser and selection state."""
    checkpoint = sim.TrainingCheckpoint(
        graph_digest=_json_digest(bundle.graph),
        training_digest=_json_digest(bundle.training),
        completed_updates=0,
        execution_protocol={"purpose": "inference_only_encoder_decoder_transfer"},
        initialization={"kind": "single_ping_transfer", "source_epoch": source_epoch},
        parameters={
            name: value.detach().cpu().clone() for name, value in parameters.items()
        },
        optimizer_state={},
        rng_state=torch.Generator().manual_seed(seed).get_state(),
        data_state={"epoch": 0, "relay_finetuning_updates": 0},
    )
    sim.save_training_checkpoint(path, checkpoint)
    return checkpoint
