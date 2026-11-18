"""Native CUDA optical flow, with optional OpenCV DIS for a controlled port comparison.

Build first: cd flowcasting_cuda; xmake f -m release; xmake
No PyTorch/CuPy dependency is required by this binding. Every returned array is on the CPU.
"""
import ctypes as ct
import os
from pathlib import Path
import threading

import numpy as np

from .optical_flow import DEFAULT_CONFIG, _crop_region, _dis, configuration as cpu_configuration


class _Config(ct.Structure):
    _fields_ = [('pairs', ct.c_int), ('gap_count', ct.c_int), ('gaps', ct.c_int * 8)] + [
        (name, ct.c_int) for name in ('align_motion', 'confidence_weighting', 'smooth_motion',
                                     'growth_decay', 'multiscale', 'midpoint', 'zero_boundary', 'scale_count')
    ] + [(name, ct.c_float) for name in ('recency_decay', 'smoothing_sigma', 'photometric_scale',
                                       'consistency_scale', 'growth_sigma', 'growth_cap', 'growth_lifetime')
    ] + [('sigmas', ct.c_float * 8), ('lifetimes', ct.c_float * 8)]


class _Timing(ct.Structure):
    _fields_ = [(name, ct.c_float) for name in ('upload_ms', 'motion_ms', 'evolution_ms', 'download_ms')]


def _native_config(config):
    if config.pairs > 8 or len(config.motion_gaps) > 8 or len(config.scale_sigmas) > 8:
        raise ValueError('CUDA supports up to eight pairs, motion gaps and spatial scales')
    result = _Config()
    for name in ('pairs', 'align_motion', 'confidence_weighting', 'smooth_motion', 'growth_decay',
                 'multiscale', 'recency_decay', 'smoothing_sigma', 'photometric_scale',
                 'consistency_scale', 'growth_sigma', 'growth_cap', 'growth_lifetime'):
        setattr(result, name, getattr(config, name))
    result.gap_count = len(config.motion_gaps)
    result.gaps = (ct.c_int * 8)(*config.motion_gaps)
    result.midpoint = config.integration == 'midpoint'
    result.zero_boundary = config.boundary == 'zero'
    result.scale_count = len(config.scale_sigmas)
    result.sigmas = (ct.c_float * 8)(*config.scale_sigmas)
    result.lifetimes = (ct.c_float * 8)(*config.scale_lifetimes)
    return result


def _pointer(array):
    return None if array is None else ct.c_void_p(array.ctypes.data)


class CudaOpticalFlow:
    """Reusable CUDA context and workspace; serialize calls to each instance.

    motion_backend='cuda' keeps the full pipeline on the GPU. 'opencv' computes the
    same CPU DIS pair fields as the reference, then performs all remaining work on CUDA.
    Both modes include input/output copies in predict(); stages expose CUDA-event times.
    """
    def __init__(self, device=0, motion_backend='cuda', library=None):
        if motion_backend not in ('cuda', 'opencv'):
            raise ValueError('motion_backend must be cuda or opencv')
        root = Path(__file__).resolve().parents[1] / 'flowcasting_cuda'
        candidates = ([Path(library)] if library else
                      [Path(os.environ['FLOWCAST_CUDA_LIBRARY'])] if 'FLOWCAST_CUDA_LIBRARY' in os.environ else
                      sorted(root.glob('build/**/flowcasting_cuda.dll')) +
                      sorted(root.glob('build/**/libflowcasting_cuda.so')))
        if not candidates:
            raise RuntimeError('CUDA library missing. Build with: cd flowcasting_cuda; xmake f -m release; xmake')
        self.library_path = candidates[0].resolve()
        self.lib = ct.CDLL(str(self.library_path))
        self.lib.fc_create.argtypes = [ct.c_int]
        self.lib.fc_create.restype = ct.c_void_p
        self.lib.fc_destroy.argtypes = [ct.c_void_p]
        self.lib.fc_destroy.restype = None
        self.lib.fc_error.restype = ct.c_char_p
        self.lib.fc_device.argtypes = [ct.c_void_p]
        self.lib.fc_device.restype = ct.c_char_p
        self.lib.fc_predict.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_int,
                                       ct.c_int, ct.POINTER(_Config), ct.c_void_p, ct.c_void_p,
                                       ct.c_void_p, ct.POINTER(_Timing)]
        self.lib.fc_predict.restype = ct.c_int
        self.lib.fc_advect.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                                      ct.c_int, ct.c_int, ct.c_int, ct.POINTER(_Config), ct.c_void_p]
        self.lib.fc_advect.restype = ct.c_int
        self.context = self.lib.fc_create(device)
        if not self.context:
            raise RuntimeError(self.lib.fc_error().decode())
        self.device_name = self.lib.fc_device(self.context).decode()
        self.device = device
        self.motion_backend = motion_backend
        self.stages = []
        self._lock = threading.Lock()

    def close(self):
        with self._lock:
            if self.context:
                self.lib.fc_destroy(self.context)
                self.context = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if getattr(self, 'context', None):
            self.lib.fc_destroy(self.context)
            self.context = None

    def configuration(self, config=DEFAULT_CONFIG):
        result = cpu_configuration(config)
        result.update(implementation='FlowCast Optical Flow CUDA', cuda_runtime=self.lib.fc_runtime_version(),
                      device=self.device_name, motion_backend=self.motion_backend,
                      solver=('parallel inverse search, four propagation passes, robust red-black refinement'
                              if self.motion_backend == 'cuda' else 'OpenCV DIS medium (CPU)'),
                      interpolation='OpenCV-compatible 1/32-pixel remapping')
        return result

    def predict(self, inputs, steps=12, *, config=DEFAULT_CONFIG, output_region=None):
        """uint8 (B,T,H,W) -> uint8 (B,steps,H,W), float32 (B,H,W,2)."""
        inputs = np.asarray(inputs)
        if (inputs.ndim != 4 or not len(inputs) or inputs.dtype != np.uint8 or inputs.shape[1] < 2
                or min(inputs.shape[2:]) < 12):
            raise ValueError('Expected nonempty uint8 (B,T,H,W), T >= 2, H/W >= 12')
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('steps must be a positive integer')
        native = _native_config(config)
        region = _crop_region(inputs.shape[2:], output_region)
        batch, t, h, w = inputs.shape
        predictions = np.empty((batch, steps, h, w), np.uint8)
        motions = np.empty((batch, h, w, 2), np.float32)
        timings = []
        with self._lock:
            if not self.context:
                raise RuntimeError('CUDA context is closed')
            for b, frames in enumerate(inputs):
                frames = np.ascontiguousarray(frames)
                supplied = None
                if self.motion_backend == 'opencv':
                    gaps = tuple(g for g in config.motion_gaps if g < t) or (1,)
                    pairs = []
                    for gap in gaps:
                        for age in range(min(config.pairs, t-gap)):
                            current, previous = frames[-1-age], frames[-1-age-gap]
                            pairs.extend((_dis().calc(current, previous, None),
                                          _dis().calc(previous, current, None)))
                    supplied = np.ascontiguousarray(np.stack(pairs))
                timing = _Timing()
                status = self.lib.fc_predict(self.context, _pointer(frames), t, h, w, steps,
                                             ct.byref(native), _pointer(supplied), _pointer(predictions[b]),
                                             _pointer(motions[b]), ct.byref(timing))
                if status:
                    raise RuntimeError(self.lib.fc_error().decode())
                timings.append({k: getattr(timing, k) for k, _ in _Timing._fields_})
            self.stages = timings
        return predictions[:, :, region[0], region[1]], motions[:, region[0], region[1]]

    def extrapolate(self, last, motion, steps=12, *, trend=None, config=DEFAULT_CONFIG):
        last, motion = np.ascontiguousarray(last, dtype=np.float32), np.ascontiguousarray(motion, dtype=np.float32)
        if (last.ndim != 2 or min(last.shape) < 12 or motion.shape != (*last.shape, 2)
                or not np.isfinite(last).all() or not np.isfinite(motion).all()):
            raise ValueError('Expected finite last (H,W), H/W >= 12 and motion (H,W,2)')
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('steps must be a positive integer')
        if trend is not None:
            trend = np.ascontiguousarray(trend, dtype=np.float32)
            if trend.shape != last.shape or not np.isfinite(trend).all():
                raise ValueError('trend must be finite and match last')
        native = _native_config(config)
        out = np.empty((steps, *last.shape), np.uint8)
        with self._lock:
            if not self.context:
                raise RuntimeError('CUDA context is closed')
            status = self.lib.fc_advect(self.context, _pointer(last), _pointer(motion), _pointer(trend),
                                        *last.shape, steps, ct.byref(native), _pointer(out))
            if status:
                raise RuntimeError(self.lib.fc_error().decode())
        return out


_local = threading.local()


def optical_flow_cuda(inputs, steps=12, *, config=DEFAULT_CONFIG, output_region=None):
    """Convenience API retaining a context per calling thread; no CPU fallback."""
    if not hasattr(_local, 'predictor'):
        _local.predictor = CudaOpticalFlow()
    return _local.predictor.predict(inputs, steps, config=config, output_region=output_region)
