"""
Backend placeholder Colorization model.

This file provides a minimal `ColorizationModel` class so the backend
can run as a skeleton. Replace this with your real model implementation
or load weights in `__init__`/a helper function.
"""

import torch
import torch.nn as nn


class ColorizationModel(nn.Module):
    """Minimal placeholder model.

    Forward: expects grayscale input tensor (B, 1, H, W) or (B, H, W)
    and returns an RGB tensor by repeating the gray channel.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has shape (B, 1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # Naive colorization: replicate grayscale channel to RGB
        return x.repeat(1, 3, 1, 1)


def load_pretrained(path: str = None):
    """Optional helper to load pretrained weights later.

    If `path` is provided, load state_dict from the given file.
    """
    model = ColorizationModel()
    if path is not None:
        state = torch.load(path, map_location='cpu')
        model.load_state_dict(state)
    return model
