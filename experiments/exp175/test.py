"""Non-simulating protocol, graph, dataset and parameter-invariant checks."""

import copy
import struct
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from experiments.exp175 import dataset, recipe
from experiments.exp175.analyse import summarize
from experiments.exp175.compute import transfer_parameters, verify_parameters
from experiments.exp175.custom.reconstruction import pixel_mse, pixel_output
from experiments.exp175.custom.snapshots import capture_selected
from experiments.exp175.custom.streaming import (
    differentiable_counts,
    native_spike_counts,
)
from experiments.exp175.custom.transfer import inference_checkpoint
from snnlab import sim
from snnlab.sim.graph import forward_pass
from snnlab.sim.graph.plan import _signal_axes
from snnlab.sim.inputs import resolution


def test_configuration_contract():
    cfg = recipe.configuration()
    assert (cfg["train_images"], cfg["validation_images"]) == (1000, 100)
    assert cfg["validation_presentations"] == 400
    assert cfg["steps"] == 2000
    assert cfg["window_steps"] == [[500, 900], [1500, 1900]]
    assert cfg["device"] == "mps" and cfg["epochs"] == 10
    assert cfg["source_cell"] == "ping__off__seed42"
    assert cfg["source_run"] == "exp175-r001-import"
    assert cfg["white_pixel_count"] == 20.0
    cfg["window_steps"][0][0] = -1
    assert recipe.configuration()["window_steps"][0][0] == 500


def test_single_ping_contains_only_one_body_and_transfers_weights(tmp_path):
    cfg = recipe.configuration(architecture="single")
    bundle = recipe.author_network(cfg)
    assert {p["id"] for p in bundle.graph["populations"]} == {"A_E", "A_I", "readout"}
    assert {i["id"] for i in bundle.graph["inputs"]} == {"X_A", "score_window"}
    projections = {p["id"]: p for p in bundle.graph["projections"]}
    assert projections["readout_projection"]["source"] == "A_E.spikes"
    assert set(bundle.training["parameter_groups"][1]["parameters"]) == set(
        recipe.frozen_parameters(cfg)
    )
    built = sim.build(
        bundle=bundle,
        training=bundle.training,
        device="cpu",
        performance=sim.Performance(compile_mode=None),
    )
    selected = {
        name: value.detach().clone()
        for name, value in built.model.parameter_map().items()
    }
    # All three relay bodies must match the single body; trainable matrices
    # are copied, while existing relay feedback/coupling remains untouched.
    base = {name: torch.zeros(1) for name in recipe.FROZEN}
    for sender in "ABC":
        for role in recipe.LOCAL_ROLES:
            base[f"{sender}_{role}.weight"] = selected[f"A_{role}.weight"].clone()
    base.update({name: torch.zeros_like(selected[name]) for name in recipe.TRAINABLE})
    for name in recipe.TRAINABLE:
        selected[name].fill_(0.123)
    transferred = transfer_parameters(base, selected)
    assert all(torch.equal(transferred[name], base[name]) for name in recipe.FROZEN)
    assert all(
        torch.equal(transferred[name], selected[name]) for name in recipe.TRAINABLE
    )
    relay = recipe.author_network(recipe.configuration())
    inference_checkpoint(
        relay, transferred, tmp_path / "transfer", seed=42, source_epoch=3
    )
    restored = sim.load_training_checkpoint(tmp_path / "transfer")
    assert restored.completed_updates == 0 and restored.optimizer_state == {}
    assert restored.data_state["relay_finetuning_updates"] == 0
    assert all(
        torch.equal(restored.parameters[name], value)
        for name, value in transferred.items()
    )


def test_single_input_matches_attended_relay_stream():
    arrays = dict(
        digits=np.array([0, 1]),
        variants=np.array([0, 3]),
        train_rows=np.array([0, 1]),
        validation_rows=np.array([0, 1]),
        ids=np.arange(2),
        features=np.zeros((2, 784), dtype=np.uint8),
    )
    arrays["features"][0, :100] = 255
    arrays["features"][1, 200:350] = 200
    context = SimpleNamespace(
        phase="simulate",
        epoch=0,
        seed=lambda sample, stream: sample + (100 if stream == "sensory_B" else 0),
    )
    cfg = recipe.configuration(device="cpu")
    options = dict(
        dt_ms=cfg["dt_ms"],
        channels=784,
        seed=cfg["seed"],
        context=context,
        device="cpu",
    )
    relay = dataset.encode(arrays, [0, 1], config={"recipe": cfg}, **options)
    single = dataset.encode(
        arrays, [0, 1], config={"recipe": dict(cfg, architecture="single")}, **options
    )
    assert set(single) == {"X_A", "score_window"}
    assert torch.equal(single["X_A"][:, 0], relay["X_A"][:, 0])
    assert torch.equal(single["X_A"][:, 1], relay["X_B"][:, 1])
    assert torch.equal(single["score_window"], relay["score_window"])


@pytest.mark.parametrize(
    "override",
    [
        {"train_images": 6301},
        {"validation_images": 701},
        {"split_seed": 43},
        {"epochs": 0},
        {"epochs": True},
        {"contrast_ratio": 1},
        {"lr": float("nan")},
        {"dt_ms": 0.2},
        {"switch_ms": 2400},
        {"windows_ms": ((1, 1300), (1900, 2000))},
        {"unknown": 1},
        {"n_e": 400},
    ],
)
def test_invalid_overrides(override):
    with pytest.raises(ValueError):
        recipe.configuration(**override)


def test_shared_encoder_and_frozen_membership():
    bundle = recipe.author_network(recipe.configuration())
    graph = bundle.graph
    parameters = {p["id"] for p in graph["parameters"]}
    assert parameters == set(recipe.TRAINABLE) | set(recipe.FROZEN)
    projections = {p["id"]: p for p in graph["projections"]}
    plan = sim.plan_graph(graph)
    assert {p.id for p in plan.projections if not p.enabled} == set(recipe.DISABLED)
    for sender in "ABC":
        for role in ("E_to_I", "I_to_E"):
            assert projections[f"{sender}_{role}"].get("enabled", True)
    assert (
        projections["sensory_A"]["parameters"]
        == projections["sensory_B"]["parameters"]
        == ["shared_encoder.weight"]
    )
    groups = bundle.training["parameter_groups"]
    assert set(groups[0]["parameters"]) == set(recipe.TRAINABLE)
    assert set(groups[1]["parameters"]) == set(recipe.FROZEN)
    assert groups[1]["frozen"]
    assert not bundle.training.get("regularizers")
    assert bundle.training["objectives"][0]["target"] == "image_index"
    assert bundle.training["objectives"][0]["definition"] == "exp175.pixel_mse/v1"
    sim.CheckpointSelection(metric=sim.SelectionMetric("loss")).validate(
        bundle.training, epochs=50, has_validation=True
    )
    operations = {op["id"]: op for op in graph["operations"]}
    assert operations["readout_counts"]["config"]["mask"] == "score_window.value"
    assert operations["readout_counts"]["kind"] == "reduce_sum"
    assert operations["readout_counts"]["sources"][0] == "readout.spikes"
    output = next(pop for pop in graph["populations"] if pop["id"] == "readout")
    assert output["size"] == 784 and output["spiking"]
    assert output["neuron"]["tau"] == {"value": 2.0, "unit": "ms"}
    assert output["neuron"]["soft_reset_threshold"] == 1.0
    assert not any(p["source"].startswith("score_window") for p in graph["projections"])


def test_native_initialisation_and_selected_snapshot_without_timesteps(tmp_path):
    cfg = recipe.configuration()
    bundle = recipe.author_network(cfg)
    built = sim.build(
        bundle=bundle,
        training=bundle.training,
        device="cpu",
        performance=sim.Performance(compile_mode=None),
    )
    params = built.model.parameter_map()
    assert {key for key, value in params.items() if value.requires_grad} == set(
        recipe.TRAINABLE
    )
    assert params["shared_encoder.weight"].shape == (784, 1024)
    selected = {key: value.detach().clone() for key, value in params.items()}
    selected["shared_encoder.weight"].fill_(0.01)
    selected["readout_projection.weight"].fill_(0.5)
    checkpoint = SimpleNamespace(
        parameters=selected, data_state={"epoch": 2}, completed_updates=50
    )
    capture_selected(bundle, checkpoint, cfg, tmp_path / "selected")
    native = sim.NetworkSnapshot.read(tmp_path / "selected", parameters=True)
    native.validate(verify_parameters=True)
    assert set(native.execution["parameter_membership"]["trainable"]) == set(
        recipe.TRAINABLE
    )
    loaded = native.load_parameters()
    assert set(loaded) == set(params)
    assert all(torch.equal(loaded[key], value) for key, value in selected.items())


def test_split_exactly_matches_tr02_and_four_validation_conditions():
    labels = np.arange(60000) % 10
    pixels = np.zeros((60000, 784), np.uint8)
    arrays = dataset.snapshot_arrays(pixels, labels)
    train, validation = dataset.source_indices(labels)
    assert len(train) == 6300 and len(validation) == 700
    assert set(train).isdisjoint(validation)
    np.testing.assert_array_equal(arrays["original_ids"][:6300], train)
    np.testing.assert_array_equal(
        arrays["original_ids"][6300:].reshape(-1, 4),
        np.repeat(validation[:, None], 4, axis=1),
    )
    np.testing.assert_array_equal(
        arrays["variants"][6300:].reshape(-1, 4), np.tile(np.arange(4), (700, 1))
    )


def test_contrast_is_mean_matched_and_fixed():
    pixels = np.zeros((2, 784), np.uint8)
    pixels[0, :100] = 255
    pixels[1, :300] = 200
    cfg = recipe.configuration()
    high, low = dataset.contrast_rates(pixels, cfg)
    np.testing.assert_allclose(high.mean(1), cfg["mean_input_hz"])
    np.testing.assert_allclose(low.mean(1), cfg["mean_input_hz"])
    np.testing.assert_allclose(
        high - cfg["mean_input_hz"], 12 * (low - cfg["mean_input_hz"]), atol=1e-12
    )
    assert high.min() >= 0 and low.min() >= 0


def test_pairing_never_crosses_split_or_loses_target():
    arrays = dict(
        digits=np.array([0, 1, 2, 3, 4, 5]),
        variants=np.array([-1, -1, 0, 1, 2, 3]),
        train_rows=np.array([0, 1]),
        validation_rows=np.array([2, 3, 4, 5]),
    )
    cfg = recipe.configuration()
    for row in range(6):
        training = row < 2
        pair, sender, phase = dataset.presentation(
            arrays, row, cfg, training=training, epoch=3
        )
        assert pair[sender] == row
        assert (
            pair[1 - sender] in arrays["train_rows" if training else "validation_rows"]
        )
        assert arrays["digits"][pair[0]] != arrays["digits"][pair[1]]
        assert phase in (0, 1)
        assert (pair, sender, phase) == dataset.presentation(
            arrays, row, cfg, training=training, epoch=3
        )
        if not training:
            assert 2 * sender + phase == arrays["variants"][row]


def test_tr02_encoder_is_copied_and_decoder_is_fresh():
    raw = {
        "W_ff.0": torch.arange(784 * 1024).reshape(784, 1024).float(),
        "W_ff.1": torch.arange(1024 * 10).reshape(1024, 10).float(),
    }
    raw.update(
        {
            "W_ee.1": torch.zeros(1024, 1024),
            "W_ei.1": torch.ones(1024, 256),
            "W_ie.1": torch.full((256, 1024), 2.0),
            "W_ii.1": torch.zeros(256, 256),
        }
    )
    cfg = recipe.configuration()
    params = recipe.initialise(raw, cfg)
    compact = recipe.initialise(
        {key: raw[key] for key in ("W_ff.0", "W_ei.1", "W_ie.1")}, cfg
    )
    assert all(torch.equal(params[key], compact[key]) for key in params)
    assert torch.equal(params["shared_encoder.weight"], raw["W_ff.0"])
    assert params["readout_projection.weight"].shape == (1024, 784)
    assert (params["readout_projection.weight"] >= 0).all()
    assert torch.equal(
        params["readout_projection.weight"],
        recipe.initialise(raw, cfg)["readout_projection.weight"],
    )
    for sender in "ABC":
        assert torch.equal(params[f"{sender}_E_to_I.weight"], raw["W_ei.1"])
        assert torch.equal(params[f"{sender}_I_to_E.weight"], raw["W_ie.1"])
    torch.testing.assert_close(
        params["A_E_to_C_E.weight"].sum(0), torch.full((1024,), cfg["relay_e_total_us"])
    )
    torch.testing.assert_close(
        params["A_E_to_C_I.weight"].sum(0), torch.full((256,), cfg["relay_i_total_us"])
    )
    assert set(params) == set(recipe.FROZEN) | set(recipe.TRAINABLE)


def test_frozen_change_and_missing_training_change_are_errors():
    initial = {key: torch.ones(1) for key in (*recipe.FROZEN, *recipe.TRAINABLE)}
    final = copy.deepcopy(initial)
    for name in recipe.TRAINABLE:
        final[name] += 1
    verify_parameters(final, initial, require_change=True)
    final[recipe.FROZEN[0]] += 1
    with pytest.raises(ValueError, match="Frozen"):
        verify_parameters(final, initial, require_change=True)
    with pytest.raises(ValueError, match="did not change"):
        verify_parameters(initial, initial, require_change=True)


def test_summary_keeps_initial_and_final_distinct_from_selected():
    rows = [
        dict(epoch=i, validation_loss=loss)
        for i, loss in ((0, 1.0), (1, 0.5), (2, 0.7))
    ]
    checks = dict(
        selected_epoch=1,
        all_frozen_weights_identical=True,
        shared_encoder=True,
        limitation="validation",
    )
    summary = summarize(dict(epochs=rows), checks)
    assert summary["initial_validation_mse"] == 1.0
    assert summary["selected_validation_mse"] == 0.5
    assert summary["final_validation_mse"] == 0.7
    assert summary["official_test_evaluated"] is False
    assert summary["attention_content_selection_verified"] is False


def test_reconstruction_loss_uses_actual_grayscale_target(tmp_path):
    images = np.zeros((2, 784), dtype=np.uint8)
    images[0, :100] = 128
    images[1, 200:400] = 255
    path = tmp_path / "images.idx"
    path.write_bytes(struct.pack(">IIII", 2051, 2, 28, 28) + images.tobytes())
    expected = torch.from_numpy(images[[1, 0]].copy()).float() / 255
    cfg = {"image_file": str(path)}
    ids = torch.tensor([1, 0])
    assert pixel_mse(expected, ids, cfg) == 0
    assert pixel_mse(expected.flip(0), ids, cfg) > 0
    prediction = torch.full_like(expected, 0.5, requires_grad=True)
    pixel_mse(prediction, ids, cfg).backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0
    pixels = pixel_output(
        (torch.tensor([[0.0, 10.0, 20.0, 30.0]]),), (), {"white_count": 20.0}
    )
    torch.testing.assert_close(pixels, torch.tensor([[0.0, 0.5, 1.0, 1.5]]))


def test_pilot_subsets_stay_inside_original_partitions():
    labels = np.arange(60000) % 10
    pixels = np.zeros((60000, 784), np.uint8)
    cfg = recipe.configuration(train_images=1000, validation_images=100)
    arrays = dataset.snapshot_arrays(pixels, labels, cfg)
    original_train, original_validation = dataset.source_indices(labels)
    train, validation = arrays["source_train"], arrays["source_validation"]
    assert len(train) == 1000 and len(validation) == 100
    assert set(train).issubset(original_train)
    assert set(validation).issubset(original_validation)
    assert set(train).isdisjoint(validation)
    assert len(arrays["features"]) == 1400
    assert len(set(labels[train])) == len(set(labels[validation])) == 10


def test_streamed_counts_match_dense_values_and_gradients_without_history():
    graph = recipe.author_network(recipe.configuration()).graph
    online = differentiable_counts(
        graph,
        _signal_axes(graph),
        {},
        enabled=False,
        window=None,
        start=0,
        steps=5,
        saved={},
        tensors={},
    )
    assert set(online.reductions) == {"readout_counts.value"}
    remaining = online.history_operations({"pixel_intensities.value"})
    assert not any("readout.spikes" in op["sources"] for op in remaining)
    spikes = torch.rand(5, 2, 784, requires_grad=True)
    mask = torch.tensor(
        [[False, False], [True, False], [True, False], [False, True], [False, True]]
    )
    for step in range(5):
        online.observe(
            {"readout.spikes": spikes[step], "score_window.value": mask[step]}, step
        )
    streamed = online.finish(5)["readout_counts.value"]
    dense = (spikes * mask[..., None]).sum(0)
    weights = torch.linspace(-1, 1, 784)
    torch.testing.assert_close(streamed, dense)
    streamed_grad = torch.autograd.grad(
        (streamed * weights).sum(), spikes, retain_graph=True
    )[0]
    dense_grad = torch.autograd.grad((dense * weights).sum(), spikes)[0]
    torch.testing.assert_close(streamed_grad, dense_grad)
    assert not streamed_grad[~mask].any()


def test_dataset_mask_adapter_is_scoped_and_zero_copy():
    original = resolution._dataset_batch_values
    original_online = forward_pass.OnlineReductions
    mask = torch.ones(5, 2, dtype=torch.bool)
    arrays = {"X_A": torch.zeros(5, 2, 784), "score_window": mask}
    with pytest.raises(ValueError, match="align"):
        original(arrays, binding=None, cap=2)
    with pytest.raises(RuntimeError, match="restore"), native_spike_counts():
        tensors, _ = resolution._dataset_batch_values(arrays, binding=None, cap=2)
        assert tensors["score_window"].shape == (5, 2)
        assert tensors["score_window"].data_ptr() == mask.data_ptr()
        assert forward_pass.OnlineReductions is differentiable_counts
        raise RuntimeError("restore")
    assert resolution._dataset_batch_values is original
    assert forward_pass.OnlineReductions is original_online


def test_native_dataset_resolves_2d_scoring_mask_without_neural_execution(tmp_path):
    cfg = recipe.configuration(device="cpu")
    pixels = np.zeros((2, 784), np.uint8)
    pixels[0, :100] = 255
    pixels[1, 200:400] = 255
    path = tmp_path / "dataset.npz"
    np.savez_compressed(
        path,
        features=pixels,
        ids=np.arange(2),
        original_ids=np.arange(2),
        digits=np.array([0, 1]),
        variants=np.full(2, -1),
        train_rows=np.arange(2),
        validation_rows=np.arange(2),
    )
    binding = replace(dataset.binding(path, cfg, "train"), sample_indices=(0,))
    bundle = recipe.author_network(cfg)
    with native_spike_counts():
        inputs, targets = sim.resolve_dataset_snapshot_binding(
            bundle.graph, binding, device="cpu"
        )
    assert inputs.tensors["score_window"].shape == (2000, 1)
    assert int(inputs.tensors["score_window"].sum()) == 400
    assert inputs.tensors["X_A"].shape == (2000, 1, 784)
    assert targets[0].value.tolist() == [0]
