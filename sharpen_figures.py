"""Before/after figures for the loss-function study (sharpen_queue.py).

For a few official test windows chosen by ground-truth storm coverage only (distinct events, as in
make_figures.py), render every method side by side: a static grid (t+15, t+30, t+60 min and a zoom on the
strongest storm at t+60) and an animation. Also collects the metrics the report charts need.

    python sharpen_figures.py --models "Earthformer=earthformer-pretrained,Earthformer + GAN=runs/sharp/ef_gan/last.pt,..."
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from flowcast import viz
from flowcast.data import IN_FRAMES, read_test_samples
from flowcast.optical_flow import optical_flow
from flowcast.predict import forecast, load_model
from make_figures import pick_cases

LEADS = (3, 6, 12)          # frames -> 15, 30, 60 min
ZOOM = 128                  # km


def zoom_box(truth, size=ZOOM):
    """Top-left corner of the size x size box holding the most intense observed storm at t+60."""
    smooth = F.avg_pool2d(torch.as_tensor(truth[-1], dtype=torch.float32)[None, None], 31, 1, 15)[0, 0].numpy()
    y, x = np.unravel_index(np.argmax(smooth), smooth.shape)
    h, w = smooth.shape
    return int(np.clip(y - size // 2, 0, h - size)), int(np.clip(x - size // 2, 0, w - size))


def grid(path, rows, box, title):
    names = list(rows)
    fig, axes = plt.subplots(len(LEADS) + 1, len(names), figsize=(2.05 * len(names), 2.1 * (len(LEADS) + 1) + 0.35),
                             squeeze=False, facecolor=viz.PLATE)
    y0, x0 = box
    for j, name in enumerate(names):
        for i in range(len(LEADS) + 1):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#2a2f37')
            if i < len(LEADS):
                frame = rows[name][LEADS[i] - 1]
                ax.imshow(frame, cmap=viz.VIL_CMAP, norm=viz.VIL_NORM, interpolation='nearest')
                if i == len(LEADS) - 1:      # mark the zoom area on the t+60 frame
                    ax.add_patch(plt.Rectangle((x0, y0), ZOOM, ZOOM, fill=False, ec='white', lw=0.8, ls='--'))
            else:
                frame = rows[name][LEADS[-1] - 1][y0:y0 + ZOOM, x0:x0 + ZOOM]
                ax.imshow(frame, cmap=viz.VIL_CMAP, norm=viz.VIL_NORM, interpolation='nearest')
            if i == 0:
                ax.set_title(name, fontsize=9.5, color=viz.PLATE_INK)
            if j == 0:
                label = f't+{LEADS[i] * 5} min' if i < len(LEADS) else f't+60 zoom ({ZOOM} km)'
                ax.set_ylabel(label, fontsize=9, color=viz.PLATE_INK)
    fig.suptitle(title, fontsize=10.5, color=viz.PLATE_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=100, facecolor=viz.PLATE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', required=True, help='"Label=source,..." in display order (optical flow is added)')
    parser.add_argument('--out', default='results/sharp/figures')
    parser.add_argument('--cases', type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 5e9 / torch.cuda.mem_get_info()[1]))

    per_sample = dict(np.load('results/sharp_cpu/per_sample_optical_flow.npz'))   # coverage = ground truth only
    cases, meta = pick_cases(per_sample, n_storms=args.cases - 1)
    data = read_test_samples(cases)                                                 # (n, 25, H, W) uint8
    inputs, truth = data[:, :IN_FRAMES], data[:, IN_FRAMES:]

    rows = {'Observed': truth.astype(np.float32), 'Optical flow': optical_flow(inputs)[0]}
    for item in args.models.split(','):
        label, _, source = item.partition('=')
        model = load_model(source.strip(), 'cuda')
        rows[label.strip()] = np.concatenate([o['forecast'].float().cpu().numpy()
                                              for o in forecast(model, inputs, batch=1)])
        del model
        torch.cuda.empty_cache()

    info = []
    for n, idx in enumerate(cases):
        case = {k: v[n] for k, v in rows.items()}
        when = str(meta.time_utc[idx])[:16]
        kind = 'storm event' if str(meta.id[idx]).startswith('S') else 'random-sample event'
        title = f'Test window {idx} ({kind}, {when} UTC): forecasts from the same 65 minutes of radar'
        grid(out / f'case{n}.png', case, zoom_box(truth[n]), title)
        viz.comparison_gif(out / f'case{n}.gif', inputs[n], {k: v for k, v in case.items()}, scale=0.42)
        info.append({'index': int(idx), 'time_utc': when, 'id': str(meta.id[idx]), 'kind': kind,
                     'cover133': float(per_sample['cover133'][idx])})
        print(f'case {n}: window {idx}', flush=True)
    (out / 'cases.json').write_text(json.dumps({'cases': info, 'methods': list(rows)}, indent=1))


if __name__ == '__main__':
    main()
