"""Build report figures and animations from results/ and the trained model.

Showcase test cases are chosen by ground-truth storm coverage only (distinct events), never by which
method scores best on them, so the visual comparison is not cherry-picked.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from flowcast import baselines, viz
from flowcast.optical_flow import optical_flow
from flowcast.pysteps import pysteps
from flowcast.data import IN_FRAMES, ROOT, read_test_samples
from flowcast.predict import forecast, load_model


PER_SAMPLE = {'FlowCastNet': 'flowcast', 'Earthformer': 'earthformer', 'Earthformer FT': 'earthformer_ft',
              'Optical flow': 'optical_flow', 'Pysteps LK': 'pysteps'}


def pick_cases(per_sample, n_storms=3, seed=0):
    meta = pd.read_csv(ROOT / 'nowcast_testing_000_META.csv')
    cover = per_sample['cover133']
    chosen, seen = [], set()
    for i in np.argsort(-cover):
        if meta.id[i] not in seen:
            chosen.append(int(i))
            seen.add(meta.id[i])
        if len(chosen) == n_storms:
            break
    moderate = [i for i in np.flatnonzero((per_sample['cover74'] > 0.03) & (cover < 0.02)) if meta.id[i] not in seen]
    chosen.append(int(np.random.default_rng(seed).choice(moderate)))
    return chosen, meta


def anatomy(path, inputs, truth, out, lead=12):
    """Show how the forecast decomposes into learned advection + growth/decay."""
    flow = out['flow'][0, lead - 1].permute(1, 2, 0).cpu().numpy()          # (H, W, 2) backward displacement
    advected = out['advected'][0, lead - 1].cpu().numpy()
    residual = out['residual'][0, lead - 1].cpu().numpy()
    fc = out['forecast'][0, lead - 1].cpu().numpy()
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), facecolor=viz.PLATE)
    panels = [(inputs[-1], 'Last observation + learned motion'), (advected, f'Advected (t+{lead * 5} min)'),
              (residual, 'Growth / decay residual'), (fc, 'FlowCastNet forecast'), (truth[lead - 1], 'Ground truth')]
    for ax, (img, title) in zip(axes, panels):
        if title.startswith('Growth'):
            lim = max(10.0, float(np.percentile(np.abs(img), 99.5)))
            im = ax.imshow(img, cmap='RdBu_r', vmin=-lim, vmax=lim)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(colors=viz.PLATE_INK, labelsize=8)
        else:
            ax.imshow(img, cmap=viz.VIL_CMAP, norm=viz.VIL_NORM, interpolation='nearest')
        ax.set_title(title, fontsize=10, color=viz.PLATE_INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(viz.PLATE)
    h, w = flow.shape[:2]
    step = 32
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    axes[0].quiver(xs, ys, -flow[ys, xs, 0], -flow[ys, xs, 1], color='white', angles='xy',
                   scale_units='xy', scale=1, width=0.006)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=viz.PLATE)
    plt.close(fig)


def rollout_curves(path, results, horizon=60):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for name, r in results.items():
        leads = np.arange(1, len(r['per_lead']['mse']) + 1) * 5
        color = viz.METHOD_COLORS.get(name)
        for ax, t in zip(axes[:2], (74, 133)):
            ax.plot(leads, r['per_lead']['csi'][0][r['thresholds'].index(t)], label=name, color=color, lw=2)
            ax.set_title(f'CSI  VIL ≥ {t}', fontsize=10)
        axes[2].plot(leads, np.array(r['per_lead']['mse']) * 1e3, label=name, color=color, lw=2)
        axes[2].set_title('MSE (×10⁻³)', fontsize=10)
    for ax in axes:
        ax.axvline(horizon, color='k', ls=':', lw=1)
        ax.text(horizon + 2, ax.get_ylim()[1] * 0.97, 'trained horizon', fontsize=8, va='top')
        ax.set_xlabel('lead time (min)')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def training_curves(path, log_path):
    rows = [json.loads(line) for line in Path(log_path).read_text().splitlines()]
    train = [r for r in rows if 'loss' in r]
    val = [r for r in rows if 'val_csi_m' in r]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    steps = np.array([r['step'] for r in train])
    loss = np.array([r['loss'] for r in train])
    axes[0].plot(steps, loss, color='#bbbbbb', lw=1)
    k = max(1, len(loss) // 40)
    axes[0].plot(steps[k - 1:], np.convolve(loss, np.ones(k) / k, 'valid'), color=viz.METHOD_COLORS['FlowCastNet'], lw=2)
    axes[0].set_title('training loss (balanced MSE+MAE)', fontsize=10)
    axes[1].plot([r['step'] for r in val], [r['val_csi_m'] for r in val], 'o-', color=viz.METHOD_COLORS['FlowCastNet'])
    axes[1].set_title('validation CSI-M (EMA weights)', fontsize=10)
    for ax in axes:
        ax.set_xlabel('step')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def scatter(path, nn, of):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, t in zip(axes, (74, 133)):
        a, b, c = of[f'csi{t}'], nn[f'csi{t}'], nn[f'cover{t}']
        keep = c > 0.005
        ax.scatter(a[keep], b[keep], s=6, c=np.log10(c[keep]), cmap='viridis', alpha=0.7)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        win = (b[keep] > a[keep]).mean()
        ax.set_title(f'per-window CSI  VIL ≥ {t}\nFlowCastNet better on {win:.0%} of {keep.sum()} windows', fontsize=10)
        ax.set_xlabel('FlowCast Optical Flow')
        ax.set_ylabel('FlowCastNet')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/flowcast/best.pt', help='FlowCastNet run (training curve)')
    parser.add_argument('--models', default='FlowCastNet=runs/flowcast/best.pt',
                        help='Comma-separated name=checkpoint pairs to show in the case figures')
    parser.add_argument('--results', default='results')
    args = parser.parse_args()
    res = Path(args.results)
    fig_dir = res / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 2.0e9 / torch.cuda.mem_get_info()[1]))

    metrics = json.loads((res / 'metrics.json').read_text())
    viz.lead_time_curves(fig_dir / 'lead_curves.png', metrics)
    training_curves(fig_dir / 'training.png', Path(args.checkpoint).parent / 'log.jsonl')
    per_sample = {n: np.load(res / f'per_sample_{k}.npz') for n, k in PER_SAMPLE.items()
                  if (res / f'per_sample_{k}.npz').exists()}
    scatter(fig_dir / 'scatter.png', per_sample['FlowCastNet'], per_sample['Optical flow'])

    models = {}
    for item in args.models.split(','):
        name, source = item.split('=', 1)
        models[name] = load_model(source)
    cases, meta = pick_cases(per_sample['Optical flow'])     # coverage depends on ground truth only
    case_info = []
    for n, idx in enumerate(cases):
        x = read_test_samples([idx])
        inputs, truth = x[:, :IN_FRAMES], x[:, IN_FRAMES:]
        rows = {'Ground truth': truth[0]}
        for name, model in models.items():
            out = forecast(model, inputs, return_parts=(name == 'FlowCastNet'))[0]
            rows[name] = np.clip(np.rint(out['forecast'][0].cpu().numpy()), 0, 255).astype(np.uint8)
            if n == 0 and name == 'FlowCastNet':
                anatomy(fig_dir / 'anatomy.png', inputs[0], truth[0], out)
        rows['Optical flow'] = optical_flow(inputs)[0][0]
        if 'Pysteps LK' in metrics:
            rows['Pysteps LK'] = pysteps(inputs)[0][0]
        rows['Persistence'] = np.repeat(inputs[0, -1:], 12, 0)
        when = meta.time_utc[idx]
        title = f'SEVIR test #{idx}  ·  {when} UTC'
        viz.forecast_grid(fig_dir / f'case{n}.png', inputs[0], rows, title=title)
        animated = {k: v for k, v in rows.items() if k != 'Persistence'}
        viz.comparison_gif(fig_dir / f'case{n}.gif', inputs[0], animated, scale=0.6 if len(animated) <= 4 else 0.5,
                           title=title)
        case_info.append({'index': idx, 'time_utc': when, 'id': meta.id[idx],
                          'csi74': {k: float(v['csi74'][idx]) for k, v in per_sample.items()}})
        print(f'case {n}: test #{idx} ({when})', flush=True)

    rollout_path = res / 'rollout_metrics.json'
    if rollout_path.exists():
        rollout_curves(fig_dir / 'rollout_curves.png', json.loads(rollout_path.read_text()))
        s = np.load(res / 'rollout_showcase.npz')
        names = [str(n) for n in s['names']] if 'names' in s else ['FlowCastNet', 'Optical flow']
        pref = ['Earthformer', 'Earthformer FT', 'FlowCastNet', 'Optical flow', 'Persistence']
        names = sorted(names, key=lambda n: pref.index(n) if n in pref else len(pref))
        rows = {'Ground truth': s['truth'], **{n: s[n.replace(' ', '_')] for n in names if n != 'Persistence'}}
        label = f'held-out event {s["id"]}'
        viz.comparison_gif(fig_dir / 'rollout.gif', s['inputs'], rows, scale=0.75 if len(rows) <= 3 else 0.5,
                           title=f'3-hour autoregressive simulation · {label}')
        viz.forecast_grid(fig_dir / 'rollout.png', s['inputs'], rows, leads=(6, 12, 18, 24, 30, 36),
                          title=f'3-hour simulation · {label}')
    (res / 'cases.json').write_text(json.dumps(case_info, indent=1))
    print('Figures written to', fig_dir.resolve())


if __name__ == '__main__':
    main()
