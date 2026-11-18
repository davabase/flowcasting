# CPU vs CUDA optical flow comparison

Measured on November 18, 2026 on an NVIDIA GeForce RTX 5060 Ti and Intel Core i9-12900K (24 logical CPUs), Windows. CUDA toolkit/runtime 13.1, Visual Studio 2022, OpenCV 5.0.0.

The CPU and CUDA implementations were scored afresh on the same 1,872 official SEVIR test windows: 13 observations and 12 forecast frames, 384 × 384 pixels. The selected V3 motion gaps and evolution settings were held fixed.

| Implementation | CSI-M ↑ | CSI ≥ 133 ↑ | CSI ≥ 219 ↑ | MSE ×10⁻³ ↓ |
|---|---:|---:|---:|---:|
| CPU (Python / OpenCV) | 0.38343 | 0.35121 | 0.13942 | 5.75543 |
| CUDA | 0.38309 | 0.35054 | 0.13828 | 5.74234 |

Full CUDA uses a separately implemented DIS solver with parallel patch propagation and refinement. It is a numerical variant rather than a bitwise OpenCV port. Its CSI-M is slightly lower and its MSE slightly lower on this test set. Its forecast MAE relative to Python is 1.61284 encoded VIL units; individual pixels can differ substantially (observed maximum 249). These are point estimates, without confidence intervals.

| Implementation | Single mean ms | Single p95 ms | Batch mean ms/window | Batch windows/s | Single speedup | Batch speedup |
|---|---:|---:|---:|---:|---:|---:|
| CPU (Python / OpenCV) | 410.23 | 431.71 | 77.44 | 12.91 | 1.00× | 1.00× |
| CUDA | 9.26 | 10.58 | 9.20 | 108.65 | 44.28× | 8.41× |

Warm timings use 32 fixed, evenly spaced test windows, three repeats, and rotating implementation order. Single-window latency uses batch 1; batch throughput uses 8 windows per call. Python uses eight workers with one OpenCV thread per worker. CUDA solves pairs and pixels in parallel and currently processes batch windows sequentially.

Wall clocks include dispatch, input upload, forecast and motion download, and synchronization. They exclude disk reads, scoring, compilation and initial context/workspace setup. GPU-event stage times, observed first-call times, raw samples, configurations, exact indices, input hashes, source hashes and per-lead metrics are stored in local `results/cuda_comparison.json`. Desktop load and hardware affect these measurements.

Build and benchmark commands, numerical differences and API usage are in [the CUDA implementation guide](../flowcasting_cuda/README.md). Run `python build_report.py` after benchmarking to regenerate `results/report/index.html`, including timing/skill plots and a matched forecast image. The historical model scorecard remains separate from this fresh paired comparison.
