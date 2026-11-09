# Neural model comparison

Completed 2026-11-08 (America/Los_Angeles). All six methods use the same 1,872 official SEVIR test windows, original 384 × 384 grid, 13 input frames, 12 target frames, and shared metrics.

| Method | CSI-M ↑ | CSI ≥ 133 ↑ | MSE ×10⁻³ ↓ |
|---|---:|---:|---:|
| Earthformer | 0.4469 | 0.4357 | 3.6364 |
| Earthformer FT | 0.4279 | 0.4311 | 3.8308 |
| FlowCastNet | 0.3966 | 0.4002 | 4.5777 |
| pySTEPS LK | 0.3507 | 0.3178 | 7.1347 |
| FlowCast Optical Flow | 0.3495 | 0.3128 | 7.1907 |
| Persistence | 0.2654 | 0.2253 | 11.3346 |

Pretrained Earthformer has the highest CSI-M. FlowCastNet improves over FlowCast Optical Flow and the pySTEPS LK baseline. Fine-tuning Earthformer does not improve test CSI-M in this run: 0.4279 versus 0.4469 for the pretrained weights.

FlowCastNet and FlowCast Optical Flow are our in-house models. FlowCast Optical Flow uses OpenCV DIS motion estimation and semi-Lagrangian extrapolation, with pySTEPS LK as the external optical flow baseline. Pretrained and fine-tuned Earthformer serve as the neural comparison baselines; persistence is a shared reference.

## Training and selection

- FlowCastNet: trained from scratch for 20,000 steps, batch 4; selected EMA checkpoint at step 20,000 (validation CSI-M 0.4181).
- Earthformer FT: fine-tuned the pretrained model for 8,000 steps, batch 1, learning rate 5e-5, frozen convolution stem, no rotation/flip augmentation; selected EMA checkpoint at step 1,000 (validation CSI-M 0.4747). The final checkpoint was not the best validation checkpoint.
- 5,146 clean training events and 155 held-out validation events. Each validation check uses the first 60 validation events (180 fixed windows). Event IDs were checked to be disjoint from the official test set.
- Seed 42, bf16, NVIDIA GeForce RTX 5060 Ti, PyTorch 2.15.0+cu132. Training was interrupted on November 4 and resumed from optimizer/EMA checkpoints on November 8; resumed sampling is not an exact replay of an uninterrupted run.

Earthformer includes external pretraining, so this comparison does not isolate architecture from training data and compute. These are single-run point estimates without confidence intervals. The classical baselines were evaluated on November 4; the models on November 8.

## Artifacts and reproduction

Open `results/report/index.html` for interactive lead-time curves, full threshold and pooled metrics, training history, and four radar forecast comparisons. Figures include all three model variants plus FlowCast Optical Flow, pySTEPS LK, and persistence. The separate three-hour rollout experiment was not run.

Full metrics: `results/metrics.json`; per-window scores: `results/per_sample_<method>.npz`; checkpoints and run configs: `runs/flowcast/` and `runs/earthformer_ft/`. Generated results, data and checkpoints follow the existing gitignore convention.

Run `python run_comparison.py` to resume unfinished configured schedules and regenerate evaluation, figures and the report. It skips completed training, rejects duplicate runners, and writes progress to `logs/comparison_status.json` and `logs/comparison_<stage>.log`.

Validation passed: completed training steps and selected checkpoint IDs match the validation logs; saved EMA tensors and all 1,872 per-window scores are finite; target coverage agrees across all methods; optical flow tests and dependency checks pass.
