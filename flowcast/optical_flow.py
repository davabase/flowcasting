"""FlowCast Optical Flow: OpenCV DIS motion and semi-Lagrangian extrapolation."""
import concurrent.futures
import threading

import cv2
import numpy as np

_local = threading.local()


def _dis():
    if not hasattr(_local, 'dis'):
        _local.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _local.dis


def backward_motion(frames, pairs=3):
    """Displacement b (H, W, 2) with frame[t](p) ~= frame[t-1](p + b(p)), averaged over recent pairs."""
    flows = [_dis().calc(frames[-1 - k], frames[-2 - k], None) for k in range(pairs)]
    return np.mean(flows, axis=0)


def extrapolate(last, motion, steps):
    """Semi-Lagrangian advection: follow backward trajectories through a stationary motion field."""
    h, w = last.shape
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = np.empty((steps, h, w), np.uint8)
    for k in range(steps):
        dx = cv2.remap(motion[..., 0], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dy = cv2.remap(motion[..., 1], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x, y = x + dx, y + dy
        out[k] = cv2.remap(last, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def optical_flow_one(frames, steps=12):
    frames = np.ascontiguousarray(frames)
    motion = backward_motion(frames)
    return extrapolate(frames[-1], motion, steps), motion


def optical_flow(inputs, steps=12, workers=8):
    """inputs uint8 (B, T_in, H, W) -> (forecast (B, steps, H, W) uint8, motion (B, H, W, 2))."""
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:   # OpenCV releases the GIL
        results = list(pool.map(lambda f: optical_flow_one(f, steps), inputs))
    return np.stack([r[0] for r in results]), np.stack([r[1] for r in results])
