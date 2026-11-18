"""Build the loss-function study page (report_sharp/index.html).

Scores come from results/sharp*/metrics.json and figures from the folders written by sharpen_figures.py. The
narrative is report_sharp/report_text.json; its {{run:metric:fmt}} placeholders are filled with live numbers.

    python build_sharp_report.py
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

from flowcast import viz
from flowcast.metrics import THRESHOLDS, summary_row
from flowcast.sharpness import sharp_row

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'report_sharp'

# Every run: label, model family, loss recipe in words.
RUNS = {
    'Persistence': ('Persistence', 'ref', 'last frame, frozen'),
    'Optical flow': ('Optical flow', 'ref', 'DIS motion + semi-Lagrangian advection'),
    'ef_original': ('Earthformer (original)', 'earthformer', 'official weights, MSE-trained'),
    'fc_original': ('FlowCastNet (original)', 'flowcast', 'weighted MSE+MAE, motion losses'),
    'ef_mse': ('Earthformer + MSE (control)', 'earthformer', 'mse'),
    'ef_spectral': ('Earthformer + radial spectrum', 'earthformer', 'mse + 0.5 radial spectrum + 0.5 gradient difference'),
    'ef_gan': ('Earthformer + PatchGAN', 'earthformer', 'mse + 0.5 PatchGAN'),
    'ef_jaccard': ('Earthformer + soft Jaccard', 'earthformer', 'mse + 1 soft Jaccard (6 thresholds)'),
    'ef_ftversky': ('Earthformer + focal Tversky', 'earthformer', 'mse + 1 focal Tversky'),
    'ef_boundary': ('Earthformer + boundary loss', 'earthformer', 'mse + 0.5 soft Jaccard + 0.5 boundary'),
    'ef_hausdorff': ('Earthformer + Hausdorff loss', 'earthformer', 'mse + 1 Hausdorff (distance transform)'),
    'ef_ssim': ('Earthformer + SSIM', 'earthformer', 'mse + 1 multi-scale SSIM'),
    'ef_split_spec': ('Earthformer, split + spectrum', 'earthformer', 'low-pass mse + 1 radial spectrum + 0.5 gradient difference'),
    'ef_split_gan': ('Earthformer, split + GAN', 'earthformer', 'low-pass mse + 0.5 radial spectrum + 1 PatchGAN'),
    'ef_split_k': ('Earthformer, split + k-space + GAN', 'earthformer', 'low-pass mse + 1 2-D k-space + 1 PatchGAN'),
    'fc_split_gan': ('FlowCastNet, split + GAN', 'flowcast', 'low-pass mse + 0.5 radial spectrum + 1 PatchGAN'),
    'fc_split_k': ('FlowCastNet, split + k-space + GAN', 'flowcast', 'low-pass mse + 1 2-D k-space + 1 PatchGAN'),
    'fc_split_wk': ('FlowCastNet, weighted split + k-space + GAN', 'flowcast', 'low-pass weighted mse + 1 2-D k-space + 1 PatchGAN'),
    'fc_split_wg': ('FlowCastNet, weighted split + GAN', 'flowcast', 'low-pass weighted mse + 1 PatchGAN'),
    'blend_k': ('Earthformer + FlowCastNet detail (k)', 'blend', 'Earthformer above 16 km, sharpened FlowCastNet (split + k-space + GAN) below'),
    'blend_wk': ('Earthformer + FlowCastNet detail (wk)', 'blend', 'Earthformer above 16 km, sharpened FlowCastNet (weighted split) below'),
    'blend_oft': ('Earthformer + radar texture', 'blend', 'Earthformer above 16 km, optical-flow-advected radar below'),
    'ef_refine': ('Earthformer + GAN refiner', 'earthformer', 'frozen Earthformer + 3.3M noise-conditioned texture GAN (one sample)'),
    'fc_refine': ('FlowCastNet + GAN refiner', 'flowcast', 'frozen FlowCastNet + 3.3M noise-conditioned texture GAN (one sample)'),
    'fc_ftversky': ('FlowCastNet + focal Tversky', 'flowcast', 'weighted mse + 1 focal Tversky'),
    'ef_tv_refine': ('Earthformer + Tversky + GAN refiner', 'earthformer', 'frozen focal-Tversky Earthformer + 3.3M texture GAN (one sample)'),
    'fc_tv_refine': ('FlowCastNet + Tversky + GAN refiner', 'flowcast', 'frozen focal-Tversky FlowCastNet + 3.3M texture GAN (one sample)'),
    'ef_tvtv_refine': ('Earthformer + Tversky, Tversky-aware refiner', 'earthformer', 'frozen focal-Tversky Earthformer + texture GAN trained with focal Tversky on the pooled mean (one sample)'),
    'fc_tvtv_refine': ('FlowCastNet + Tversky, Tversky-aware refiner', 'flowcast', 'frozen focal-Tversky FlowCastNet + texture GAN trained with focal Tversky on the pooled mean (one sample)'),
}
COLOR = {'ref': '--ref', 'earthformer': '--s4', 'flowcast': '--s1', 'blend': '--s5'}
REF_COLOR = {'Persistence': '--s3', 'Optical flow': '--s2'}


def load_results():
    out = {}
    for name in ('sharp_cpu', 'sharp'):
        path = ROOT / 'results' / name / 'metrics.json'
        if path.exists():
            out.update(json.loads(path.read_text()))
    return {k: v for k, v in out.items() if k in RUNS}


def row(r):
    return {**summary_row(r), **sharp_row(r)}


def color_of(key):
    return REF_COLOR.get(key) or COLOR[RUNS[key][1]]


_PLACEHOLDER = re.compile(r'\{\{([^:{}]+):([^:{}]+):([^{}]+)\}\}')


def fill(obj, rows):
    """Replace {{run:metric:fmt}} in narrative text with live numbers ('pct' = percent), recursively."""
    def one(m):
        run, metric, fmt = m.groups()
        if run not in rows:
            return '…'
        v = rows[run][metric]
        if fmt == 'pct':
            return f'{v * 100:.0f}%' if v >= 0.1 else f'{v * 100:.1f}%'
        return format(v, fmt)
    if isinstance(obj, str):
        return _PLACEHOLDER.sub(one, obj)
    if isinstance(obj, list):
        return [fill(o, rows) for o in obj]
    if isinstance(obj, dict):
        return {k: fill(v, rows) for k, v in obj.items()}
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    res = load_results()
    rows = {k: row(v) for k, v in res.items()}
    text = fill(json.loads((OUT / 'report_text.json').read_text(encoding='utf-8')), rows)
    hl = text['highlight']                       # which runs the charts and tiles feature
    labels = {k: lab for k, lab, _ in hl['spectrum']}
    picks = set(hl['picks'])                      # labelled on the scatter, highlighted in the table

    # --- spectrum (bins 1..N/2 -> wavelengths 384..2 km), leads 15/30/60 min
    wl = [w for w in res['ef_original']['sharp']['wavelength_km'][1:]]
    leads = [3, 6, 12]
    dash = {'solid': '', 'dashed': '5 5', 'dotted': '1.5 4'}
    spectrum = {'wavelength_km': [round(w, 3) for w in wl], 'leads': [l * 5 for l in leads], 'series': [
        {'label': lab, 'color': color_of(k), 'dash': dash[style],
         'ratio': [[round(float(v), 5) for v in np.asarray(res[k]['sharp']['psd_ratio'][l - 1][1:])] for l in leads]}
        for k, lab, style in hl['spectrum'] if k in res]}

    # --- trade-off scatter
    metrics = [
        {'key': 'csim', 'label': 'CSI-M', 'note': 'CSI-M: pixel-exact hit rate averaged over six VIL thresholds. It rewards smooth forecasts (double penalty).'},
        {'key': 'pool16', 'label': 'CSI-M, 16 km', 'note': 'CSI-M after max-pooling both fields to 16 km cells: credit for getting the storm in the right neighbourhood.'},
        {'key': 'fss', 'label': 'FSS 133 @ 17 km', 'note': 'Fractions skill score for heavy rain (VIL ≥ 133) in 17 × 17 km windows. 1 is perfect; it does not double-penalise displaced detail.'},
        {'key': 'csi219', 'label': 'CSI-219', 'note': 'CSI for the most intense cores (VIL ≥ 219). Blurring erases these first.'},
        {'key': 'mse', 'label': 'MSE (×10⁻³)', 'invert': True, 'fmt': 'mse', 'note': 'Mean squared error on VIL/255, ×10⁻³. Lower is better; plotted downwards so up is always better.'},
    ]
    points = []
    for k, r in rows.items():
        points.append({'key': k, 'label': RUNS[k][0], 'recipe': RUNS[k][2], 'color': color_of(k),
                       'short': labels.get(k, RUNS[k][0]), 'highlight': k in picks,
                       'rank': hl['picks'].index(k) if k in picks else 99,
                       'detail': r['SmallPow60'], 'csim': r['CSI-M'], 'pool16': r['CSI-M-pool16'],
                       'fss': r['FSS133@17km'], 'csi219': r['CSI-219'], 'mse': r['MSE(1e-3)']})
    groups = [{'label': 'Earthformer runs', 'color': '--s4'}, {'label': 'FlowCastNet runs', 'color': '--s1'},
              {'label': 'Scale blends', 'color': '--s5'}, {'label': 'Optical flow', 'color': '--s2'},
              {'label': 'Persistence', 'color': '--s3'}]
    arrows = [a for a in hl['arrows'] if a[0] in rows and a[1] in rows]

    # --- scorecard
    cols = [('CSI-M', 'csim', 'Mean CSI over thresholds 16-219 (higher is better)', '{:.3f}', max),
            ('CSI-219', 'csi219', 'CSI for the most intense cores', '{:.3f}', max),
            ('CSI-M 16 km', 'pool16', 'CSI-M after 16 km max-pooling', '{:.3f}', max),
            ('FSS-133 17 km', 'fss', 'Fractions skill score, heavy rain, 17 km windows', '{:.3f}', max),
            ('MSE ×10⁻³', 'mse', 'Mean squared error on VIL/255 (lower is better)', '{:.2f}', min),
            ('Eff. res. 60 min', 'eff', 'Shortest wavelength (km) keeping half the observed power', '{:.0f} km', min),
            ('Detail 60 min', 'detail', 'Power at 4-16 km vs real radar', None, None)]
    order = [('Reference', ['Persistence', 'Optical flow', 'ef_original', 'fc_original']),
             ('Earthformer fine-tunes (4,000 windows each, same schedule)',
              ['ef_mse', 'ef_spectral', 'ef_gan', 'ef_jaccard', 'ef_ftversky', 'ef_boundary', 'ef_hausdorff', 'ef_ssim',
               'ef_split_spec', 'ef_split_gan', 'ef_split_k']),
             ('FlowCastNet fine-tunes (20,000 windows each, same schedule)', ['fc_ftversky', 'fc_split_gan', 'fc_split_k', 'fc_split_wk', 'fc_split_wg']),
             ('Generative refiner on the frozen model (12,000 windows each)', ['ef_refine', 'ef_tv_refine', 'ef_tvtv_refine', 'fc_refine', 'fc_tv_refine', 'fc_tvtv_refine']),
             ('Scale blends (no training)', ['blend_oft', 'blend_k', 'blend_wk'])]
    model_rows = [k for _, ks in order[1:] for k in ks if k in rows] + ['ef_original', 'fc_original']
    best = {}
    for label, key, _, _, fn in cols:
        if fn:
            vals = {k: (rows[k]['EffRes60(km)'] if key == 'eff' else next(p[key] for p in points if p['key'] == k)) for k in model_rows}
            best[key] = fn(vals, key=vals.get)
    table_groups = []
    for glabel, keys in order:
        grows = []
        for k in keys:
            if k not in rows:
                continue
            p = next(p for p in points if p['key'] == k)
            cells = {}
            for label, key, _, fmtstr, _ in cols:
                if key == 'eff':
                    cells[key] = fmtstr.format(rows[k]['EffRes60(km)'])
                elif key == 'detail':
                    v = p['detail']
                    cells[key] = f'{v * 100:.0f}%' if v >= 0.1 else f'{v * 100:.1f}%'
                else:
                    cells[key] = fmtstr.format(p[key])
            grows.append({'label': RUNS[k][0], 'recipe': RUNS[k][2], 'color': color_of(k), 'cells': cells,
                          'pick': k in hl['picks'],
                          'best': [key for key, b in best.items() if b == k]})
        if grows:
            table_groups.append({'label': glabel, 'rows': grows})

    # --- figure sets (each: a folder from sharpen_figures.py + captions), shown with a toggle
    figsets = []
    for fs in text['figsets']:
        src, dst = ROOT / fs['dir'], OUT / 'figures' / fs['key']
        dst.mkdir(parents=True, exist_ok=True)
        meta = json.loads((src / 'cases.json').read_text())
        cases = []
        for n, c in enumerate(meta['cases']):
            for ext in ('png', 'gif'):
                shutil.copy(src / f'case{n}.{ext}', dst / f'case{n}.{ext}')
            cases.append({'index': c['index'], 'label': text['case_labels'][n], 'png': f"figures/{fs['key']}/case{n}.png",
                          'gif': f"figures/{fs['key']}/case{n}.gif", 'caption': fs['captions'][n]})
        figsets.append({'label': fs['label'], 'cases': cases})

    def pct(v):
        return f'{v * 100:.0f}%' if v >= 0.1 else f'{v * 100:.1f}%'
    facts = []
    names = {'CSI-M': 'CSI-M', 'CSI-219': 'CSI-219', 'CSI-M-pool16': '16 km CSI', 'FSS133@17km': 'FSS'}
    for title, before, after, *shown in hl['facts']:      # optional 4th item: which skill scores to show
        if after not in rows or (before and before not in rows):
            continue
        shown = shown[0] if shown else ['CSI-M', 'FSS133@17km']
        a = rows[after]
        if before:
            b = rows[before]
            facts.append([title, f"{pct(b['SmallPow60'])} → {pct(a['SmallPow60'])}", ' · '.join(
                ['fine detail at 60 min'] + [f"{names[m]} {b[m]:.3f} → {a[m]:.3f}" for m in shown])])
        else:
            facts.append([title, pct(a['SmallPow60']), ' · '.join(
                ['fine detail at 60 min'] + [f"{names[m]} {a[m]:.3f}" for m in shown])])
    data = {
        'text': text['text'], 'lists': text['lists'], 'facts': facts, 'repro': text['repro'], 'footer': text['footer'],
        'vil': {'levels': viz.VIL_LEVELS, 'colors': viz.VIL_COLORS[:-1]}, 'thresholds': list(THRESHOLDS),
        'figsets': figsets, 'spectrum': spectrum,
        'trade': {'metrics': metrics, 'points': points, 'groups': groups, 'arrows': arrows},
        'table': {'cols': [{'label': c[0], 'key': c[1], 'help': c[2]} for c in cols], 'groups': table_groups},
        'losses': text['losses'],
    }
    html = (OUT / 'template.html').read_text(encoding='utf-8').replace(
        '{{DATA}}', json.dumps(data, ensure_ascii=False).replace('</', '<\\/'))
    (OUT / 'index.html').write_text(html, encoding='utf-8')
    print(f'wrote {OUT / "index.html"} ({len(html) / 1e3:.0f} kB) with {len(points)} runs')


if __name__ == '__main__':
    main()
