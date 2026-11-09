# Pysteps optical flow comparison

Evaluated on 2026-11-04 using all 1,872 official SEVIR nowcast test windows (June–November 2019). Each method uses the same 13 input frames, 12 future targets, original 384 × 384 grid, and shared scoring code.

| Method | CSI-M ↑ | CSI ≥ 74 ↑ | CSI ≥ 133 ↑ | MSE ×10⁻³ ↓ |
|---|---:|---:|---:|---:|
| Persistence | 0.2654 | 0.4656 | 0.2253 | 11.3346 |
| Optical flow | 0.3495 | 0.5869 | 0.3128 | 7.1907 |
| Pysteps LK | 0.3507 | 0.5884 | 0.3178 | 7.1347 |

Pysteps improves CSI-M slightly over the existing DIS baseline (+0.0012), with slightly lower MSE. This is a point estimate on this test set; no statistical significance claim is made.

## Method

Pysteps 1.21.5 Lucas–Kanade motion estimation uses the last four observations (three frame pairs), followed by deterministic semi-Lagrangian extrapolation with linear interpolation, one midpoint iteration, and zero fill outside the domain. Outputs are rounded to uint8. DIS uses the same three recent pairs and its existing advection implementation. All methods operate on encoded SEVIR VIL (0–255); MSE uses values divided by 255. CSI counts are aggregated over windows, pixels, and leads before division; CSI-M averages the six standard SEVIR thresholds.

This evaluates optical flow extrapolation, not the stochastic STEPS ensemble algorithm. There is no rain-rate conversion, log transform, training, or tuning on test targets.

The neural models were subsequently trained and evaluated on the same test windows. See [the full model comparison](model_comparison.md) for their results and validation-selected checkpoints.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe evaluate.py --methods persistence,optical_flow,pysteps --device cpu
.\.venv\Scripts\python.exe build_report.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Generated report: `results/report/index.html`, with lead-time curves and a forecast comparison selected solely by greatest target coverage at VIL ≥ 133. Full metrics: `results/metrics.json`; per-window scores: `results/per_sample_<method>.npz`. These generated artifacts follow the existing gitignore convention.

Validation: empty-field and known-translation tests passed; real SEVIR forecasts were identical with one and two workers; all three methods produced 1,872 finite per-window scores.

Sources: [pysteps optical flow](https://pysteps.readthedocs.io/en/stable/auto_examples/plot_optical_flow.html), [semi-Lagrangian API](https://pysteps.readthedocs.io/en/stable/generated/pysteps.extrapolation.semilagrangian.extrapolate.html).
