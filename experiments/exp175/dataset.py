"""TR02 image split and sample-stable, mean-matched contrast presentations."""

import hashlib
import struct

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from snnlab import extensions, sim

ENCODER = "pinglab.exp175.fixed_contrast/v1"


def source_indices(labels):
    pool = np.random.RandomState(42).choice(len(labels), 7000, replace=False)
    return train_test_split(pool, test_size=700, random_state=42, stratify=labels[pool])


def read_training(raw):
    if not all(
        (raw / name).is_file()
        for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    ):
        from torchvision.datasets import MNIST

        MNIST(root=raw.parents[1], train=True, download=True)
    image_bytes = (raw / "train-images-idx3-ubyte").read_bytes()
    label_bytes = (raw / "train-labels-idx1-ubyte").read_bytes()
    if struct.unpack(">IIII", image_bytes[:16]) != (2051, 60000, 28, 28):
        raise ValueError("Expected official MNIST training image IDX file")
    if struct.unpack(">II", label_bytes[:8]) != (2049, 60000):
        raise ValueError("Expected official MNIST training label IDX file")
    if len(image_bytes) != 16 + 60000 * 784 or len(label_bytes) != 8 + 60000:
        raise ValueError("Truncated MNIST training files")
    pixels = np.frombuffer(image_bytes, np.uint8, offset=16).reshape(60000, 784)
    labels = np.frombuffer(label_bytes, np.uint8, offset=8).astype(np.int64)
    if labels.min() < 0 or labels.max() > 9:
        raise ValueError("Invalid MNIST labels")
    pins = {
        "images": hashlib.sha256(image_bytes).hexdigest(),
        "labels": hashlib.sha256(label_bytes).hexdigest(),
    }
    return pixels, labels, pins


def snapshot_arrays(pixels, labels, cfg=None):
    train, validation = source_indices(labels)
    if cfg is not None:
        if cfg["train_images"] < len(train):
            train, _ = train_test_split(
                train,
                train_size=cfg["train_images"],
                random_state=42,
                stratify=labels[train],
            )
        if cfg["validation_images"] < len(validation):
            validation, _ = train_test_split(
                validation,
                train_size=cfg["validation_images"],
                random_state=42,
                stratify=labels[validation],
            )
    originals = np.concatenate((train, np.repeat(validation, 4)))
    return dict(
        ids=np.arange(len(originals)),
        original_ids=originals,
        features=pixels[originals],
        digits=labels[originals],
        variants=np.concatenate(
            (np.full(len(train), -1), np.tile(np.arange(4), len(validation)))
        ),
        train_rows=np.arange(len(train)),
        validation_rows=np.arange(len(train), len(originals), 4),
        source_train=train,
        source_validation=validation,
    )


def contrast_rates(pixels, cfg):
    """Exact scratch pair-offset encoding: both senders/cue phases share one mean."""
    images = np.asarray(pixels, np.float64) / 255.0
    means = images.mean(1)
    if means.max() <= 0:
        raise ValueError("A pair cannot contain two blank images")
    mean = cfg["mean_input_hz"]
    alpha = mean / means.max()
    high = mean + alpha * (images - means[:, None])
    low = mean + (high - mean) / cfg["contrast_ratio"]
    if high.min() < -1e-10 or low.min() < -1e-10:
        raise ValueError("Negative rate")
    if high.max() * cfg["dt_ms"] / 1000 > 1:
        raise ValueError("Bernoulli spike probability exceeds one")
    return np.maximum(high, 0), np.maximum(low, 0)


def presentation(arrays, row, cfg, *, epoch=0, training=False):
    """Target identity is fixed; distractors and cue placement vary only in training."""
    rng = np.random.default_rng(
        [cfg["seed"], 11 if training else 12, epoch if training else 0, int(row)]
    )
    pool = arrays["train_rows" if training else "validation_rows"]
    eligible = pool[arrays["digits"][pool] != arrays["digits"][row]]
    if not len(eligible):
        raise ValueError("Need a different-digit distractor inside the same split")
    partner = int(rng.choice(eligible))
    variant = int(rng.integers(4)) if training else int(arrays["variants"][row])
    if not 0 <= variant < 4:
        raise ValueError(
            "Evaluation requires one of four target-location/phase conditions"
        )
    sender, phase = variant // 2, variant % 2
    pair = [partner, partner]
    pair[sender] = int(row)
    return pair, sender, phase


def encode(arrays, selected, *, dt_ms, channels, config, seed, context, device="cpu"):
    cfg = config["recipe"]
    if dt_ms != cfg["dt_ms"] or channels != cfg["n_input"]:
        raise ValueError("Encoder and graph dimensions/timestep differ")
    n, steps, switch = len(selected), cfg["steps"], cfg["switch_step"]
    single = cfg.get("architecture", "relay") == "single"
    drives = {
        f"X_{s}": torch.empty((steps, n, 784), dtype=torch.float32, device=device)
        for s in ("A" if single else "AB")
    }
    mask = torch.zeros((steps, n), dtype=torch.bool, device=device)
    for batch, row in enumerate(selected):
        pair, target_sender, target_phase = presentation(
            arrays,
            int(row),
            cfg,
            epoch=context.epoch if context.phase == "train" else 0,
            training=int(arrays["variants"][row]) < 0,
        )
        high, low = contrast_rates(arrays["features"][pair], cfg)
        first_high_sender = target_sender if target_phase == 0 else 1 - target_sender
        for sender, name in enumerate("AB"):
            if single and sender != target_sender:
                continue
            output_name = "A" if single else name
            stream_seed = context.seed(int(arrays["ids"][row]), f"sensory_{name}")
            random_device = "cpu" if torch.device(device).type == "mps" else device
            generator = torch.Generator(device=random_device).manual_seed(stream_seed)
            for phase, (start, stop) in enumerate(((0, switch), (switch, steps))):
                rates = (
                    high[sender]
                    if sender
                    == (first_high_sender if phase == 0 else 1 - first_high_sender)
                    else low[sender]
                )
                probability = torch.tensor(
                    rates * dt_ms / 1000, dtype=torch.float32, device=random_device
                )
                drives[f"X_{output_name}"][start:stop, batch] = (
                    (
                        torch.rand(
                            (stop - start, 784),
                            generator=generator,
                            device=random_device,
                        )
                        < probability
                    )
                    .float()
                    .to(device)
                )
        start, stop = cfg["window_steps"][target_phase]
        mask[start:stop, batch] = True
    if not single:
        drives["X_C"] = torch.zeros((1, n, 1024), device=device).expand(steps, n, 1024)
    drives["score_window"] = mask
    return drives


def encoder(cfg):
    registered = next(
        (d for d in extensions.definitions("encoder") if d.name == ENCODER), None
    )
    if registered is None:
        extensions.register_encoder(ENCODER, encode)
    elif registered.function is not encode:
        raise ValueError("Encoder definition collision")
    return sim.DatasetEncoder(
        "custom",
        definition=ENCODER,
        seed=cfg["seed"],
        rng_policy="sample_v1",
        duration_ms=cfg["duration_ms"],
        config={"recipe": cfg},
    )


def binding(path, cfg, split):
    start, stop = (
        (0, cfg["train_images"])
        if split == "train"
        else (
            cfg["train_images"],
            cfg["train_images"] + cfg["validation_presentations"],
        )
    )
    return sim.DatasetSnapshotBinding(
        path,
        "X_A",
        "exp175-mnist",
        split,
        encoder(cfg),
        target_id="image_index",
        label_key="original_ids",
        sample_ids_key="ids",
        sample_indices=tuple(range(start, stop)),
    )
