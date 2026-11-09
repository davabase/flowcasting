"""External pySTEPS baseline: Lucas–Kanade motion and deterministic extrapolation."""
import concurrent.futures

import numpy as np


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
