"""Plotting helpers: SEVIR-style VIL palette, forecast grids, animated comparisons, metric curves."""
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image, ImageDraw, ImageFont

# Encoded-VIL bins and colours following the SEVIR tutorial palette (black = no echo).
VIL_LEVELS = [0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 256]
VIL_COLORS = ['#000000', '#4d4d4d', '#28be28', '#199619', '#0a690a', '#0a4b0a',
              '#f5f500', '#edac00', '#f06e00', '#a00000', '#e700ff']
VIL_CMAP = ListedColormap(VIL_COLORS)
VIL_NORM = BoundaryNorm(VIL_LEVELS, VIL_CMAP.N)
_LUT = (np.array([matplotlib.colors.to_rgb(VIL_COLORS[np.searchsorted(VIL_LEVELS, v, 'right') - 1])
                  for v in range(256)]) * 255).astype(np.uint8)

METHOD_COLORS = {'FlowCastNet': '#2a78d6', 'Optical flow': '#eb6834', 'Persistence': '#1baf7a',
                 'Earthformer': '#6250d6', 'Earthformer FT': '#e87ba4', 'Pysteps LK': '#b58b12'}
PLATE = '#111418'   # dark 'radar scope' background for imagery figures
PLATE_INK = '#e6e9ee'


def colorize(frame):
    """uint8 (H, W) encoded VIL -> RGB uint8 (H, W, 3)."""
    return _LUT[np.clip(np.asarray(frame), 0, 255).astype(np.uint8)]


def _font(size):
    for name in ('arialbd.ttf', 'DejaVuSans-Bold.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def comparison_gif(path, inputs, rows, step_minutes=5, scale=0.5, title=None):
    """Animate inputs then forecasts. rows: {name: (T_out, H, W)}; the first row should be ground truth."""
    names = list(rows)
    h, w = inputs.shape[1:]
    th, tw = int(h * scale), int(w * scale)
    pad, label_h = 6, 26
    header = 34 if title else 0
    canvas_w = len(names) * (tw + pad) + pad
    canvas_h = header + label_h + th + pad + 22
    font, small = _font(15), _font(13)
    images = []

    def tile(frame):
        return Image.fromarray(colorize(frame)).resize((tw, th), Image.NEAREST)

    total = inputs.shape[0] + rows[names[0]].shape[0]
    for k in range(total):
        canvas = Image.new('RGB', (canvas_w, canvas_h), (18, 18, 22))
        draw = ImageDraw.Draw(canvas)
        if title:
            draw.text((pad, 8), title, fill=(235, 235, 235), font=font)
        observed = k < inputs.shape[0]
        minutes = (k - inputs.shape[0] + 1) * step_minutes
        for j, name in enumerate(names):
            x0 = pad + j * (tw + pad)
            frame = inputs[k] if observed else rows[name][k - inputs.shape[0]]
            canvas.paste(tile(frame), (x0, header + label_h))
            draw.text((x0, header + 4), name if not observed else ('Observed' if j == 0 else ''),
                      fill=(235, 235, 235), font=font)
        stamp = f'observed  t{minutes:+d} min' if observed else f'forecast  t+{minutes} min'
        draw.text((pad, header + label_h + th + 4), stamp,
                  fill=(150, 200, 255) if observed else (255, 190, 120), font=small)
        images.append(canvas)
    durations = [120] * inputs.shape[0] + [260] * (total - inputs.shape[0])
    durations[-1] = 1500
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)


def forecast_grid(path, inputs, rows, leads=(1, 3, 6, 9, 12), step_minutes=5, title=None, flow=None):
    """Static figure: last observation, then selected lead times for each method (rows)."""
    names = list(rows)
    cols = 1 + len(leads)
    fig, axes = plt.subplots(len(names), cols, figsize=(2.3 * cols, 2.35 * len(names) + 0.5),
                             squeeze=False, facecolor=PLATE)
    for i, name in enumerate(names):
        for j in range(cols):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(PLATE)
            for spine in ax.spines.values():
                spine.set_color('#2a2f37')
            if j == 0:
                if i == 0:
                    ax.imshow(inputs[-1], cmap=VIL_CMAP, norm=VIL_NORM, interpolation='nearest')
                    ax.set_title('Last observed (t0)', fontsize=9, color=PLATE_INK)
                    if flow is not None:
                        _quiver(ax, flow)
                else:
                    ax.axis('off')
                ax.set_ylabel(name, fontsize=10, color=PLATE_INK)
                continue
            lead = leads[j - 1]
            ax.imshow(rows[name][lead - 1], cmap=VIL_CMAP, norm=VIL_NORM, interpolation='nearest')
            if i == 0:
                ax.set_title(f't0 + {lead * step_minutes} min', fontsize=9, color=PLATE_INK)
    for i, name in enumerate(names):
        axes[i, 1].set_ylabel(name, fontsize=10, color=PLATE_INK)
    if title:
        fig.suptitle(title, fontsize=11, color=PLATE_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=PLATE)
    plt.close(fig)


def _quiver(ax, flow, step=32):
    """Overlay a motion field given as backward displacement (H, W, 2): arrows show -d (direction of motion)."""
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    u, v = -flow[ys, xs, 0], -flow[ys, xs, 1]
    ax.quiver(xs, ys, u, v, color='white', angles='xy', scale_units='xy', scale=0.25, width=0.006)


def lead_time_curves(path, results, step_minutes=5, thresholds=(16, 74, 133, 181, 219)):
    """results: {method: Scores.result()}. CSI per lead time for several thresholds plus MSE."""
    fig, axes = plt.subplots(1, len(thresholds) + 1, figsize=(3.1 * (len(thresholds) + 1), 3.2))
    for name, r in results.items():
        leads = np.arange(1, len(r['per_lead']['mse']) + 1) * step_minutes
        color = METHOD_COLORS.get(name)
        for ax, t in zip(axes, thresholds):
            k = r['thresholds'].index(t)
            ax.plot(leads, r['per_lead']['csi'][0][k], label=name, color=color, lw=2)
            ax.set_title(f'CSI  VIL ≥ {t}', fontsize=10)
        axes[-1].plot(leads, np.array(r['per_lead']['mse']) * 1e3, label=name, color=color, lw=2)
    axes[-1].set_title('MSE (×10⁻³)', fontsize=10)
    for ax in axes:
        ax.set_xlabel('lead time (min)')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
