# Native CUDA optical flow

This library accelerates the selected FlowCast optical-flow pipeline. The full CUDA mode
uploads observations once, keeps intermediate arrays on the GPU, and downloads uint8
forecasts and float32 backward motion. There is no neural model and no CPU optical-flow
fallback in this mode. Python provides configuration, array validation and the C ABI binding.

## Build

Requires xmake, the CUDA toolkit, a compatible NVIDIA driver, and a supported C++ compiler.
On the development Windows machine, CUDA 13.1 needs Visual Studio 2022 selected explicitly;
the default Visual Studio 2026 compiler is rejected by nvcc.

```powershell
cd flowcasting_cuda
xmake f -m release --vs=2022
xmake
cd ..
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_optical_flow_cuda.py -v
```

On Linux omit `--vs=2022`. `add_cugencodes("native")` targets the local GPU; configure
appropriate architectures in `xmake.lua` when distributing binaries to other GPUs.
The Python binding finds the release library under `flowcasting_cuda/build/`. Override
with `FLOWCAST_CUDA_LIBRARY` or the constructor's `library` argument. Build products are
ignored by Git. No Python CUDA package is needed by the binding itself.

```python
from flowcast.optical_flow_cuda import CudaOpticalFlow

with CudaOpticalFlow() as predictor:
    prediction, backward_motion = predictor.predict(inputs, steps=12)
    print(predictor.stages)  # CUDA-event milliseconds, per window
```

Inputs are uint8 `(B,T,H,W)` with at least two observations and H/W at least 12. Outputs
have the same convention as `flowcast.optical_flow.optical_flow`. Both support the selected
configuration, short histories and output cropping after full-domain prediction. CUDA
supports up to eight pairs, gaps and detail scales. Each instance owns its stream and a
reused device workspace; calls to one instance are serialized. Batch windows currently run
sequentially, while pair estimators, patches and pixels run in parallel. Use `close()` or a
context manager to release the workspace. A missing library/GPU raises an error.

`optical_flow_cuda(inputs)` is a convenience wrapper with a cached context per host thread.
`evaluate.py --methods optical_flow_cuda` also exposes the GPU method through standard scoring.

## What runs on the GPU

| Stage | Full CUDA | OpenCV DIS + CUDA control |
|---|---|---|
| Area image pyramids, Sobel gradients | CUDA | OpenCV CPU for DIS; CUDA for confidence |
| Multi-gap backward/forward pair motion | CUDA inverse search and refinement | Original OpenCV DIS medium on CPU |
| Confidence, residuals, history alignment | CUDA | CUDA |
| Reliability smoothing, masked trend medians, gap fusion | CUDA | CUDA |
| Gaussian/Laplacian bands, damped growth | CUDA | CUDA |
| Euler/midpoint trajectories, remapping, uint8 output | CUDA | CUDA |

The motion solver follows the DIS structure: 8×8 patches, stride 3, mean-normalized inverse
compositional search, coarse-to-fine initialization, residual-weighted densification, and
robust brightness/gradient refinement. It uses four double-buffered neighbor-propagation
passes with six inverse-search iterations per pass. OpenCV medium uses two ordered scans
with twelve iterations per scan. The CUDA refinement uses five fixed-point iterations and
five red-black relaxation sweeps with a separately implemented derivative/smoothness stencil.
These schedules and stencils differ from OpenCV; this is a numerical GPU variant, not a
bitwise DIS port. Its accuracy must be measured separately.

Postprocessing preserves the reference's 1/32-pixel OpenCV remap coordinates, reflect-101
Sobel boundaries, replicated Gaussian boundaries, gap normalization, masked medians,
confidence fallbacks, and ties-to-even output rounding. Floating-point reductions and
interpolation boundaries can still create output differences. The control mode isolates
these from differences in estimated pair motion:

```python
with CudaOpticalFlow(motion_backend='opencv') as control:
    prediction, motion = control.predict(inputs)
```

## Reproduce the comparison

```powershell
.\.venv\Scripts\python.exe benchmark_optical_flow_cuda.py
.\.venv\Scripts\python.exe build_report.py
```

The default run recomputes all three methods on the same 1,872 official test windows. It
then measures single-window latency and eight-window throughput on 32 fixed, evenly spaced
test windows, with three repeats. Warm wall time starts with NumPy observations and ends
with NumPy forecasts and motion; transfer and synchronization costs are included. Disk reads,
verification metrics and build/context setup are excluded. Python has eight window workers
and one OpenCV thread per worker. Method order rotates across repeats. First-call/context
times, GPU stage events, raw timing samples, source/library hashes, input hashes, exact test
indices, hardware, full parameters and per-lead metrics are recorded in
`results/cuda_comparison.json`. Existing model metrics are not overwritten.

The [recorded full-test comparison](../report/optical_flow_cuda.md) summarizes the development
machine's measured accuracy and timings. The generated HTML report includes the detailed tables,
lead-time curves and matched forecast image.

For a short smoke run:

```powershell
.\.venv\Scripts\python.exe benchmark_optical_flow_cuda.py --quality-windows 16 --timing-windows 4 --repeats 2 --out results/cuda_smoke
```

The report labels subset comparisons explicitly. Do not compare its subset scores with
the historical full-test scorecard. Cold allocation, desktop GPU load, batch size and CPU
worker count affect timings; GPU-event intervals do not include the control mode's CPU DIS.

Algorithm references: [OpenCV DIS API](https://docs.opencv.org/4.x/de/d4f/classcv_1_1DISOpticalFlow.html),
[OpenCV DIS implementation](https://github.com/opencv/opencv/blob/4.x/modules/video/src/dis_flow.cpp),
and [OpenCV variational refinement](https://github.com/opencv/opencv/blob/4.x/modules/video/src/variational_refinement.cpp).
