# FlowCast Optical Flow

The selected implementation is self-contained in `flowcast/optical_flow.py`. It uses OpenCV DIS,
encoded SEVIR VIL (0–255), and no trained weights. Motion is backward (x, y) displacement in
pixels per input interval: five minutes for SEVIR. The external baseline is `flowcast/pysteps.py`.

## Selected algorithm (version 3)

1. Independently estimate DIS motion over 2-, 3- and 6-frame gaps (10, 15 and 30 minutes),
   using up to three recent pairs per gap. Normalize displacements and intensity residuals by the gap.
2. Transport older estimates along observed trajectories onto the latest grid. Weight them by
   reconstruction error, forward/backward consistency, texture and recency (0.75 per interval).
   Smooth uncertain vectors using reliable neighbors; fall back to recency-weighted motion without support.
3. Average the three gap-specific motion fields and take the median of their intensity trends.
   Trends use aligned residual medians, spatial smoothing and a cap of ±3 encoded units per interval.
4. Advect the latest observation and trend using backward Euler trajectories. Sample every lead from
   the original fields. The trend contribution saturates with a three-interval lifetime.
5. Split the latest image at Gaussian widths 2 and 8 pixels; damp the detail bands with lifetimes
   of 36 and 108 intervals. Preserve the coarse field, replicate boundaries, and round/clip to uint8.

Short histories use supported gaps, falling back to adjacent pairs. Inputs must be uint8
`(B, T, H, W)`; outputs are forecasts `(B, steps, H, W)` and motion `(B, H, W, 2)`.
Thread-local DIS instances are reset before reuse so small images cannot alter subsequent settings.

## Selection and recorded results

Discovery used 20 held-out validation events (60 windows). Multi-gap fusion was then confirmed on
95 separate validation events (285 windows), at starts 0, 12 and 24. Compared with adjacent-pair V2,
CSI-M increased from 0.36204 to 0.37298, CSI ≥ 133 from 0.32832 to 0.34593, and MSE ×10⁻³
fell from 7.25253 to 7.00664. MSE improved in 68 of 95 events. Selection preceded official testing.

The recorded official test results cover the same 1,872 SEVIR windows:

| Predictor | CSI-M ↑ | CSI ≥ 133 ↑ | MSE ×10⁻³ ↓ |
|---|---:|---:|---:|
| FlowCast Optical Flow V3 | 0.38343 | 0.35121 | 5.75543 |
| Previous V2 | 0.36950 | 0.3324 | 5.99175 |
| pySTEPS LK | 0.3507 | 0.3178 | 7.1347 |

These are previously measured point estimates, not newly rerun evaluations. Raw metrics and
per-window scores remain in local `results/`; that directory is excluded from Git. Optional motion
ensembles, alternative backends and learned/satellite hybrids were excluded from this implementation.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe evaluate.py --methods persistence,optical_flow,pysteps --device cpu
.\.venv\Scripts\python.exe build_report.py
```

`--optical-flow-config PATH.json` accepts partial parameter overrides; `configs/optical_flow_v3.json`
records the selected preset. Metrics record resolved parameters, implementation version and OpenCV
version. Figure generation restores those parameters. V2 metadata remains supported; unversioned
scores require reevaluation because historical boundary settings cannot be inferred reliably.
`legacy_config()` supports historical interpolation checks.

For actual larger observations, pass `output_region=(top, left, height, width)` to crop after
motion estimation and advection. Replicated boundaries cannot reconstruct unobserved incoming storms
and may stretch echoes. Motion errors can resemble growth; smoothing and damping can suppress small
intense cells. Compare thresholds and lead times as well as aggregate CSI-M and MSE.
