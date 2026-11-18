"""FlowCast Optical Flow: aligned DIS motion, damped evolution and multiscale advection.

Motion is backward displacement in pixels per input interval (5 minutes for SEVIR).
Values remain encoded VIL, not rain rate. No trained weights or future frames are used.
"""
import concurrent.futures
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import threading

import cv2
import numpy as np

_local = threading.local()
VERSION = 3


@dataclass(frozen=True)
class OpticalFlowConfig:
    pairs: int = 3
    motion_gaps: tuple = (2, 3, 6)  # independent estimates over 10, 15 and 30 minutes
    integration: str = 'euler'
    align_motion: bool = True
    confidence_weighting: bool = True
    recency_decay: float = 0.75
    smooth_motion: bool = True
    smoothing_sigma: float = 3.0
    photometric_scale: float = 15.0  # encoded VIL units
    consistency_scale: float = 1.5  # pixels, forward/backward cycle error
    growth_decay: bool = True
    growth_sigma: float = 2.0
    growth_cap: float = 3.0  # maximum VIL change per interval
    growth_lifetime: float = 3.0  # intervals; trend saturates instead of growing forever
    multiscale: bool = True
    scale_sigmas: tuple = (2.0, 8.0)  # spatial Gaussian widths in pixels
    scale_lifetimes: tuple = (36.0, 108.0)  # intervals, finest band first
    boundary: str = 'edge'

    def __post_init__(self):
        for name in ('motion_gaps', 'scale_sigmas', 'scale_lifetimes'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not isinstance(self.pairs, int) or self.pairs < 1:
            raise ValueError('pairs must be a positive integer')
        if (not self.motion_gaps or any(not isinstance(v, int) or v < 1 for v in self.motion_gaps)
                or tuple(sorted(set(self.motion_gaps))) != self.motion_gaps):
            raise ValueError('motion_gaps must contain distinct increasing positive integers')
        if self.integration not in ('euler', 'midpoint'):
            raise ValueError('integration must be euler or midpoint')
        if not 0 < self.recency_decay <= 1:
            raise ValueError('recency_decay must be in (0, 1]')
        for name in ('smoothing_sigma', 'photometric_scale', 'consistency_scale',
                     'growth_sigma', 'growth_cap', 'growth_lifetime'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if len(self.scale_sigmas) != len(self.scale_lifetimes) or not self.scale_sigmas:
            raise ValueError('scale_sigmas and scale_lifetimes must have the same nonzero length')
        if any(not np.isfinite(v) or v <= 0 for v in (*self.scale_sigmas, *self.scale_lifetimes)):
            raise ValueError('Scale widths and lifetimes must be finite and positive')
        if any(b <= a for a, b in zip(self.scale_sigmas, self.scale_sigmas[1:])):
            raise ValueError('scale_sigmas must increase from fine to coarse')
        if any(b < a for a, b in zip(self.scale_lifetimes, self.scale_lifetimes[1:])):
            raise ValueError('Coarser scales must persist at least as long as finer scales')
        if self.boundary not in ('edge', 'zero'):
            raise ValueError("boundary must be 'edge' or 'zero'")


DEFAULT_CONFIG = OpticalFlowConfig()
V2_CONFIG = OpticalFlowConfig(motion_gaps=(1,), integration='euler')


def legacy_config(*, boundary='zero'):
    """Original DIS average, for reproducible comparisons with earlier results."""
    return OpticalFlowConfig(motion_gaps=(1,), integration='euler',
                             align_motion=False, confidence_weighting=False,
                             recency_decay=1.0, smooth_motion=False,
                             growth_decay=False, multiscale=False, boundary=boundary)


def load_config(path=None):
    """Load partial overrides, or the configuration recorded by evaluate.py."""
    if path is None:
        return DEFAULT_CONFIG
    values = json.loads(Path(path).read_text(encoding='utf-8'))
    if 'version' in values and 'parameters' in values:
        return config_from_result({'configuration': values})
    return OpticalFlowConfig(**values.get('parameters', values))


def config_from_result(result):
    """Render saved scores using their recorded settings; do not guess old settings."""
    recorded = result.get('configuration', {})
    if 'parameters' in recorded:
        if recorded.get('version') not in (2, VERSION):
            raise ValueError('Unsupported optical flow version; reevaluate before rendering figures')
        parameters = dict(recorded['parameters'])
        # Earlier deterministic V3 records included unused ensemble controls.
        if parameters.pop('ensemble_space', 'none') != 'none':
            raise ValueError('Experimental ensemble scores require their archived implementation')
        parameters.pop('ensemble_speed_spread', None)
        parameters.pop('ensemble_angle_spread', None)
        if recorded['version'] == 2:
            parameters.update(motion_gaps=(1,), integration='euler')
        return OpticalFlowConfig(**parameters)
    # Both zero-filled and edge-filled runs predate configuration recording.
    raise ValueError('Optical flow scores have no recorded configuration; rerun '
                     'evaluate.py --methods optical_flow before generating figures')


def configuration(config=DEFAULT_CONFIG):
    return {'implementation': 'FlowCast Optical Flow', 'version': VERSION,
            'opencv_version': cv2.__version__, 'parameters': asdict(config),
            'input_scale': 'encoded SEVIR VIL 0-255', 'output': 'rounded uint8'}


def _dis():
    if not hasattr(_local, 'dis'):
        _local.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    # DIS can lower this internally for small images; restore it before reuse.
    _local.dis.setFinestScale(1)
    return _local.dis


def _grid(shape):
    h, w = shape
    return np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))


def _sample(field, x, y, boundary='edge'):
    border = cv2.BORDER_REPLICATE if boundary == 'edge' else cv2.BORDER_CONSTANT
    return cv2.remap(field, x, y, cv2.INTER_LINEAR, borderMode=border, borderValue=0)


def _blur(field, sigma):
    return cv2.GaussianBlur(field, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)


def _inside(x, y, shape):
    h, w = shape
    return (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)


def _pair_confidence(current, previous, backward, forward, config):
    x, y = _grid(current.shape)
    sx, sy = x + backward[..., 0], y + backward[..., 1]
    error = np.abs(current.astype(np.float32) - _sample(previous.astype(np.float32), sx, sy))
    cycle_error = np.linalg.norm(backward + _sample(forward, sx, sy), axis=-1)
    image = current.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3) / 8
    texture = np.sqrt(_blur(gx * gx + gy * gy, 2.0))
    confidence = (1 / (1 + (error / config.photometric_scale) ** 2)
                  / (1 + (cycle_error / config.consistency_scale) ** 2)
                  * (0.05 + 0.95 * texture / (texture + 3.0)))
    return confidence * _inside(sx, sy, current.shape)


def _motion_history(frames, config, gap=1):
    """Transport pair estimates and intensity residuals onto the latest frame's grid."""
    frames = np.asarray(frames)
    if frames.ndim != 3 or len(frames) < 2 or frames.dtype != np.uint8:
        raise ValueError('frames must be uint8 (T, H, W) with at least two observations')
    if min(frames.shape[1:]) < 12:
        raise ValueError('DIS motion estimation needs images at least 12 pixels wide and high')
    frames = np.ascontiguousarray(frames)
    x, y = _grid(frames.shape[1:])
    available = np.ones(frames.shape[1:], dtype=bool)
    flows, confidence, residuals, valid = [], [], [], []
    for age in range(min(config.pairs, len(frames) - gap)):
        current, previous = frames[-1 - age], frames[-1 - age - gap]
        full_flow = _dis().calc(current, previous, None)
        flow = full_flow / gap
        if config.confidence_weighting:
            forward = _dis().calc(previous, current, None)
            weight = _pair_confidence(current, previous, full_flow, forward,
                                      replace(config, consistency_scale=config.consistency_scale * gap))
        else:
            weight = np.ones(current.shape, np.float32)
        # Evolution residuals always follow trajectories, including motion ablations.
        gx, gy = _grid(current.shape)
        px, py = gx + full_flow[..., 0], gy + full_flow[..., 1]
        residual = (current.astype(np.float32) - _sample(previous.astype(np.float32), px, py)) / gap
        mask = available & _inside(x, y, current.shape) & (_sample(
            _inside(px, py, current.shape).astype(np.float32), x, y) >= 0.999)
        transported = _sample(flow, x, y)
        flows.append(transported if config.align_motion else flow)
        confidence.append(_sample(weight, x, y) * mask if config.align_motion else weight)
        residuals.append(_sample(residual, x, y))
        valid.append(mask)
        available = mask
        x, y = x + transported[..., 0], y + transported[..., 1]
    return np.stack(flows), np.stack(confidence), np.stack(residuals), np.stack(valid)


def _combine_motion(flows, confidence, config):
    recency = config.recency_decay ** np.arange(len(flows), dtype=np.float32)
    weights = confidence * recency[:, None, None]
    support = weights.sum(axis=0)
    fallback = np.average(flows, axis=0, weights=recency).astype(np.float32)
    mean = (flows * weights[..., None]).sum(axis=0) / np.maximum(support[..., None], 1e-6)
    mean = np.where(support[..., None] > 1e-6, mean, fallback)
    if config.smooth_motion:
        reliable = support / recency.sum()
        norm = _blur(reliable, config.smoothing_sigma)
        smooth = _blur(mean * reliable[..., None], config.smoothing_sigma) / np.maximum(norm[..., None], 1e-6)
        smooth = np.where(norm[..., None] > 1e-6, smooth, fallback)
        trust = np.clip(reliable / 0.5, 0, 1)[..., None]
        mean = trust * mean + (1 - trust) * smooth
    return mean.astype(np.float32)


def backward_motion(frames, pairs=None, *, config=DEFAULT_CONFIG):
    """Estimate latest-grid backward displacement, aligned and weighted across recent pairs."""
    if pairs is not None:
        config = replace(config, pairs=pairs)
    return _components(frames, config)[0]


def _components(frames, config):
    gaps = tuple(gap for gap in config.motion_gaps if gap < len(frames)) or (1,)
    motions, trends = [], []
    for gap in gaps:
        flows, confidence, residuals, valid = _motion_history(frames, config, gap)
        motions.append(_combine_motion(flows, confidence, config))
        if config.growth_decay:
            trends.append(_growth_trend(residuals, valid, config))
    motion = np.mean(motions, axis=0).astype(np.float32)
    trend = np.median(trends, axis=0).astype(np.float32) if trends else None
    return motion, trend


def _growth_trend(residuals, valid, config):
    # Median rejects isolated registration errors; blur preserves broad evolution.
    masked = np.ma.array(residuals, mask=~valid)
    trend = np.ma.median(masked, axis=0).filled(0).astype(np.float32)
    coverage = valid.mean(axis=0).astype(np.float32)
    trend = _blur(trend * coverage, config.growth_sigma) / np.maximum(
        _blur(coverage, config.growth_sigma), 1e-6)
    return np.clip(trend, -config.growth_cap, config.growth_cap)


def _bands(field, sigmas):
    """Laplacian bands that sum to the input, plus a persistent coarse field."""
    bands, previous = [], field
    for sigma in sigmas:
        coarse = _blur(field, sigma)
        bands.append(previous - coarse)
        previous = coarse
    return bands, previous


def extrapolate(last, motion, steps, *, trend=None, config=None):
    """Advect once from the observation at each lead, avoiding repeated image blurring.

    With config=None this is plain advection with replicated boundaries. With a
    config, fine-scale detail decays sooner and an optional intensity trend is
    damped in time. The coarse band persists. Only the final output is quantized.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    source = np.asarray(last)
    last = np.asarray(last, dtype=np.float32)
    motion = np.asarray(motion, dtype=np.float32)
    if last.ndim != 2 or motion.shape != (*last.shape, 2):
        raise ValueError('Expected last (H, W) and motion (H, W, 2)')
    if not np.isfinite(last).all() or not np.isfinite(motion).all():
        raise ValueError('last and motion must be finite')
    if trend is not None:
        trend = np.asarray(trend, dtype=np.float32)
        if trend.shape != last.shape or not np.isfinite(trend).all():
            raise ValueError('trend must be finite and match last')
    boundary = config.boundary if config else 'edge'
    bands, coarse = _bands(last, config.scale_sigmas) if config and config.multiscale else ([], last)
    x, y = _grid(last.shape)
    out = np.empty((steps, *last.shape), np.uint8)
    for k in range(steps):
        dx = _sample(motion[..., 0], x, y)
        dy = _sample(motion[..., 1], x, y)
        if config and config.integration == 'midpoint':
            mx, my = x + 0.5 * dx, y + 0.5 * dy
            dx = _sample(motion[..., 0], mx, my)
            dy = _sample(motion[..., 1], mx, my)
        x, y = x + dx, y + dy
        # Keep OpenCV's original uint8 interpolation when all evolution is off,
        # so legacy configurations reproduce the historical scores exactly.
        image = source if not bands and trend is None and source.dtype == np.uint8 else coarse
        field = _sample(image, x, y, boundary)
        for band, lifetime in zip(bands, config.scale_lifetimes if bands else ()):
            field += np.exp(-(k + 1) / lifetime) * _sample(band, x, y, boundary)
        if config and config.growth_decay and trend is not None:
            elapsed = config.growth_lifetime * -np.expm1(-(k + 1) / config.growth_lifetime)
            field += elapsed * _sample(trend, x, y, boundary)
        out[k] = np.clip(np.rint(field), 0, 255).astype(np.uint8)
    return out


def _crop_region(shape, output_region):
    """Region is (top, left, height, width) inside a supplied larger observation."""
    if output_region is None:
        return slice(None), slice(None)
    if len(output_region) != 4 or any(not isinstance(v, int) for v in output_region):
        raise ValueError('output_region must contain four integers: top, left, height, width')
    top, left, height, width = output_region
    if min(top, left) < 0 or min(height, width) < 1 or top + height > shape[0] or left + width > shape[1]:
        raise ValueError('output_region must lie within the supplied observations')
    return slice(top, top + height), slice(left, left + width)


def optical_flow_one(frames, steps=12, *, config=DEFAULT_CONFIG, output_region=None):
    """Forecast on the entire supplied domain, optionally cropping after advection."""
    region = _crop_region(frames.shape[-2:], output_region)
    motion, trend = _components(frames, config)
    prediction = extrapolate(frames[-1], motion, steps, trend=trend, config=config)
    return prediction[:, region[0], region[1]], motion[region[0], region[1]]


def optical_flow(inputs, steps=12, workers=8, *, config=DEFAULT_CONFIG, output_region=None):
    """uint8 (B, T, H, W) -> forecasts (B, steps, H, W), backward motion (B, H, W, 2).

    output_region allows a real observation halo; edge padding is not extra data.
    Each feature can be disabled through OpticalFlowConfig for validation ablations.
    """
    inputs = np.asarray(inputs)
    if inputs.ndim != 4 or not len(inputs) or workers < 1:
        raise ValueError('Expected nonempty (B, T, H, W) inputs and positive workers')
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda f: optical_flow_one(
            f, steps, config=config, output_region=output_region), inputs))
    return np.stack([r[0] for r in results]), np.stack([r[1] for r in results])
