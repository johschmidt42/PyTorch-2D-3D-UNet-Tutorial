from typing import Callable

import numpy as np
import torch


def predict(
    img: np.ndarray,
    model: torch.nn.Module,
    preprocess: Callable,
    postprocess: Callable,
    device: str,
) -> np.ndarray:
    model.eval()
    img = preprocess(img)  # preprocess image
    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network

    out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
    result = postprocess(out_softmax)  # postprocess outputs

    return result
