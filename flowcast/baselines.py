"""Persistence baseline: repeat the latest observation."""
import numpy as np


def persistence(inputs, out_frames=12):
    """inputs (B, T_in, H, W) -> last frame repeated (B, out_frames, H, W)."""
    return np.repeat(inputs[:, -1:], out_frames, axis=1)
