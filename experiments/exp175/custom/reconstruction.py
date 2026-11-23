"""Pixel readout and image-reference targets for SNNLab's scalar-label dataset API."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from snnlab import extensions


def pixel_output(sources, parameters, config):
    return sources[0] / config["white_count"]


@lru_cache(maxsize=1)
def target_images(path):
    values = Path(path).read_bytes()
    if len(values) < 16 or int.from_bytes(values[:4], "big") != 2051:
        raise ValueError("Expected MNIST image IDX targets")
    count, height, width = (
        int.from_bytes(values[i : i + 4], "big") for i in (4, 8, 12)
    )
    if (height, width) != (28, 28) or len(values) != 16 + count * 784:
        raise ValueError("Invalid MNIST target images")
    return (
        torch.from_numpy(
            np.frombuffer(values, np.uint8, offset=16).copy().reshape(count, 784)
        ).float()
        / 255
    )


def pixel_mse(prediction, target, config):
    images = target_images(config["image_file"])
    ids = target.detach().long().cpu()
    expected = images[ids].to(device=prediction.device, dtype=prediction.dtype)
    return torch.nn.functional.mse_loss(prediction, expected)


extensions.register_operation("exp175.pixel_output/v1", pixel_output)
extensions.register_objective("exp175.pixel_mse/v1", pixel_mse, classification=False)
