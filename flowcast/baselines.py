"""Classical nowcasting baselines: persistence and optical-flow (semi-Lagrangian) extrapolation."""
import concurrent.futures
import threading

import cv2
import numpy as np

_local = threading.local()


def persistence(inputs, out_frames=12):
    """inputs (B, T_in, H, W) -> last frame repeated (B, out_frames, H, W)."""
    return np.repeat(inputs[:, -1:], out_frames, axis=1)


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


def pysteps_one(frames, steps=12):
    """Pysteps LK + semi-Lagrangian extrapolation on encoded VIL, without rain-rate conversion.

    Four recent frames supply the same three motion pairs as the DIS baseline.
    Velocity is forward (x, y) pixels per 5-minute step; outside-domain values are zero.
    This is deterministic extrapolation, not the stochastic STEPS ensemble algorithm.
    """
    from pysteps.motion import get_method
    from pysteps.extrapolation.semilagrangian import extrapolate as advect

    recent = np.ascontiguousarray(frames[-4:], dtype=np.float32)
    velocity = get_method('LK')(recent)
    prediction = advect(recent[-1], velocity, steps, outval=0.0,
                        interp_order=1, n_iter=1)
    if not np.isfinite(prediction).all() or not np.isfinite(velocity).all():
        raise ValueError('Pysteps returned non-finite forecasts or motion')
    # Match the existing DIS baseline's integer forecast representation.
    return np.clip(np.rint(prediction), 0, 255).astype(np.uint8), np.moveaxis(velocity, 0, -1)


def pysteps(inputs, steps=12, workers=4):
    """Return forecasts (B, steps, H, W) and forward motion (B, H, W, 2)."""
    # Import before starting workers so module initialization is not timed per window.
    import pysteps as package  # noqa: F401

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda f: pysteps_one(f, steps), inputs))
    return np.stack([r[0] for r in results]), np.stack([r[1] for r in results])
