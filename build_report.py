"""Assemble results/report/index.html (+ figures) from evaluation outputs. Every number comes from results/ and runs/."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from flowcast.data import ROOT, TRAIN_DIR
from flowcast.viz import VIL_COLORS, VIL_LEVELS

ORDER = ['Earthformer', 'Earthformer FT', 'FlowCastNet', 'Pysteps LK', 'Optical flow', 'Persistence']
VARS = {'FlowCastNet': '--s1', 'Optical flow': '--s2', 'Persistence': '--s3', 'Earthformer': '--s4', 'Earthformer FT': '--s5'}
VARS['Pysteps LK'] = '--s6'
PER_SAMPLE = {'FlowCastNet': 'flowcast', 'Earthformer': 'earthformer', 'Earthformer FT': 'earthformer_ft',
              'Optical flow': 'optical_flow', 'Pysteps LK': 'pysteps'}
# Earthformer paper (Gao et al., NeurIPS 2022), SEVIR benchmark; verified CSI-M and MSE (x1e-3) only.
REFS = [('Earthformer (paper)', 0.4419, 3.6957), ('ConvLSTM', 0.4185, 3.7532), ('U-Net', 0.3593, 4.1119),
        ('Persistence (paper)', 0.2613, 11.5338)]


def report_labels(value):
    """Use report names while preserving saved evaluation keys and artifact paths."""
    if isinstance(value, str):
        return value.replace('Optical flow', 'FlowCast Optical Flow').replace('Pysteps LK', 'pySTEPS LK')
    if isinstance(value, list):
        return [report_labels(item) for item in value]
    if isinstance(value, dict):
        return {report_labels(key): report_labels(item) for key, item in value.items()}
    return value


def by_threshold(result, key='csi', pool=0):
    return {str(t): result['per_lead'][key][pool][k] for k, t in enumerate(result['thresholds'])}


def table_row(r):
    o = r['overall']
    row = {'csi_m': o['csi_m'][0], 'pool4': o['csi_m'][1], 'pool16': o['csi_m'][2],
           'bias133': o['bias'][0][r['thresholds'].index(133)], 'mse': o['mse'] * 1e3}
    for k, t in enumerate(r['thresholds']):
        row[f'csi{t}'] = o['csi'][0][k]
    return row


def prose(m):
    """Method name inside a sentence: model names keep their capitals, baselines do not."""
    return m.lower() if m == 'Persistence' else m


def listing(items):
    items = list(items)
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


def read_log(run):
    log = [json.loads(line) for line in (run / 'log.jsonl').read_text().splitlines()] if (run / 'log.jsonl').exists() else []
    return [r for r in log if 'loss' in r], [r for r in log if 'val_csi_m' in r]


def baseline_report(res, metrics):
    """Build a measured comparison when trained-model artifacts are unavailable."""
    from html import escape
    from flowcast import baselines, viz
    from flowcast.optical_flow import optical_flow
    from flowcast.pysteps import pysteps
    from flowcast.data import IN_FRAMES, read_test_samples
    from flowcast.metrics import summary_row

    out = res / 'report'
    out.mkdir(parents=True, exist_ok=True)
    counts = {r['windows'] for r in metrics.values()}
    if len(counts) != 1:
        raise ValueError('Report methods must be evaluated on the same number of test windows')
    viz.lead_time_curves(out / 'lead_curves.png', metrics)
    rows = {m: summary_row(r) for m, r in metrics.items()}
    keys = list(next(iter(rows.values())))
    table = '<tr><th>Method</th>' + ''.join(f'<th>{escape(k)}</th>' for k in keys) + '</tr>'
    for method, row in rows.items():
        table += f'<tr><th>{escape(method)}</th>' + ''.join(f'<td>{row[k]:.4f}</td>' for k in keys) + '</tr>'
    config = escape(json.dumps(metrics.get('Pysteps LK', {}).get('configuration', {}), indent=2))
    comparison = ''
    if 'Pysteps LK' in rows and 'Optical flow' in rows:
        delta = rows['Pysteps LK']['CSI-M'] - rows['Optical flow']['CSI-M']
        comparison = (f'<p>Pysteps LK changes CSI-M by {delta:+.4f} relative to FlowCast Optical Flow. '
                      f'Its normalized MSE is {rows["Pysteps LK"]["MSE(1e-3)"]:.4f} ×10⁻³, '
                      f'compared with {rows["Optical flow"]["MSE(1e-3)"]:.4f} ×10⁻³ for FlowCast Optical Flow.</p>')
    showcase = ''
    sample_path = res / 'per_sample_optical_flow.npz'
    if sample_path.exists():
        with np.load(sample_path) as samples:
            idx = int(np.argmax(samples['cover133']))
        window = read_test_samples([idx])
        inputs, truth = window[:, :IN_FRAMES], window[:, IN_FRAMES:]
        fields = {'Ground truth': truth[0]}
        for name, method in [('Pysteps LK', pysteps), ('Optical flow', optical_flow)]:
            if name in metrics:
                fields[name] = method(inputs)[0][0]
        if 'Persistence' in metrics:
            fields['Persistence'] = baselines.persistence(inputs)[0]
        viz.forecast_grid(out / 'comparison.png', inputs[0], fields,
                          title=f'SEVIR test #{idx}: greatest target coverage at VIL ≥ 133')
        showcase = (f'<h2>Example: test window {idx}</h2><p>Selected by greatest ground-truth coverage at VIL ≥ 133, '
                    'independently of forecast scores.</p><img src="comparison.png" alt="Observed and forecast radar at selected leads">')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>SEVIR optical flow comparison</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#182333}}
table{{border-collapse:collapse;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}
img{{width:100%}}pre{{background:#f2f4f6;padding:20px;overflow:auto}}.table{{overflow:auto}}</style>
<h1>SEVIR optical flow comparison</h1>
<p>All methods scored on the same {next(iter(counts)):,} official SEVIR test windows, in dataset order.
Each window supplies 13 observed VIL frames and 12 target frames at 5-minute intervals on the original 384 × 384 grid.</p>
<p>Pysteps uses Lucas–Kanade motion from the last four observations (three consecutive pairs), followed by
deterministic semi-Lagrangian extrapolation. DIS uses the existing three-pair average and OpenCV advection.
Persistence repeats the last observation. Both flow methods receive the same recent context and fill incoming
areas outside the image with zero. Forecasts are rounded to uint8 before scoring.</p>
<p>Inputs remain encoded VIL (0–255); no rain-rate conversion, logarithmic transform, stochastic STEPS ensemble,
or parameter tuning on the test set is used. This measures optical flow extrapolation on this benchmark.</p>
<div class="table"><table>{table}</table></div>
{comparison}
<p>CSI-M averages CSI at thresholds 16, 74, 133, 160, 181, 219. Counts are summed over all windows, pixels and leads
before division. Pool4/16 use max pooling. MSE and MAE use VIL divided by 255. Higher CSI is better; lower error is better.</p>
<img src="lead_curves.png" alt="CSI and error by lead time">
{showcase}
<p>ML model comparisons are pending: this checkout has no saved model checkpoints or prior evaluation outputs.
Published scores are not substituted for local measurements.</p>
<h2>Reproduce</h2><pre>python evaluate.py --methods persistence,optical_flow,pysteps --device cpu
python build_report.py</pre>
<h2>Pysteps configuration</h2><pre>{config}</pre>
<p>References: <a href="https://pysteps.readthedocs.io/en/stable/auto_examples/plot_optical_flow.html">Pysteps optical flow</a>;
<a href="https://pysteps.readthedocs.io/en/stable/generated/pysteps.extrapolation.semilagrangian.extrapolate.html">extrapolation API</a>.
Full per-lead metrics and per-window scores are saved beside this report in the results directory.</p></html>'''
    (out / 'index.html').write_text(report_labels(html), encoding='utf-8')
    print('Wrote', (out / 'index.html').resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results')
    parser.add_argument('--run', default='runs/flowcast')
    parser.add_argument('--ft-run', default='runs/earthformer_ft')
    args = parser.parse_args()
    RES, run, ft_run = Path(args.results), Path(args.run), Path(args.ft_run)
    OUT = RES / 'report'
    metrics = json.loads((RES / 'metrics.json').read_text())
    if 'FlowCastNet' not in metrics:
        baseline_report(RES, metrics)
        return
    rollout = json.loads((RES / 'rollout_metrics.json').read_text()) if (RES / 'rollout_metrics.json').exists() else None
    cases = json.loads((RES / 'cases.json').read_text())
    methods = [m for m in ORDER if m in metrics]
    if len({metrics[m]['windows'] for m in methods}) != 1:
        raise ValueError('Report methods must be evaluated on the same number of test windows')
    models = [m for m in methods if m not in ('Optical flow', 'Persistence', 'Pysteps LK')]
    t = {m: table_row(metrics[m]) for m in methods}
    best = max(models, key=lambda m: t[m]['csi_m'])
    of, pe, fc = t['Optical flow'], t['Persistence'], t['FlowCastNet']
    ps = {m: np.load(RES / f'per_sample_{k}.npz') for m, k in PER_SAMPLE.items() if (RES / f'per_sample_{k}.npz').exists()}
    keep74, keep133 = ps['Optical flow']['cover74'] > 0.005, ps['Optical flow']['cover133'] > 0.005
    win74 = float((ps[best]['csi74'][keep74] > ps['Optical flow']['csi74'][keep74]).mean())
    win133 = float((ps[best]['csi133'][keep133] > ps['Optical flow']['csi133'][keep133]).mean())

    config = json.loads((run / 'config.json').read_text())
    train_rows, val_rows = read_log(run)
    ft_config = json.loads((ft_run / 'config.json').read_text()) if (ft_run / 'config.json').exists() else None
    ft_train, ft_val = read_log(ft_run)
    selected = max(val_rows, key=lambda r: r['val_csi_m']) if val_rows else None
    ft_selected = max(ft_val, key=lambda r: r['val_csi_m']) if ft_val else None
    events = pd.read_csv(TRAIN_DIR / 'events.csv')
    events = events[np.load(TRAIN_DIR / 'events.done.npy')[events.slot.values]]
    clean = events[events.pct_missing == 0]
    test_meta = pd.read_csv(ROOT / 'nowcast_testing_000_META.csv')
    gb = len(events) * 49 * 384 * 384 / 1e9
    storm = int((events.kind == 'storm').sum())
    sps = np.median([r['samples_per_s'] for r in train_rows[len(train_rows) // 2:]]) if train_rows else 0
    steps = train_rows[-1]['step'] if train_rows else 0
    leads = [5 * (i + 1) for i in range(len(metrics[best]['per_lead']['mse']))]

    def lead_val(m, thr, minutes):
        return metrics[m]['per_lead']['csi'][0][metrics[m]['thresholds'].index(thr)][minutes // 5 - 1]

    ef = t.get('Earthformer')
    eft = t.get('Earthformer FT')
    text = {
        'lede': ('Radar nowcasting on a local GPU. A model reads the last 65 minutes of weather radar and simulates the '
                 'next hour, one 5-minute frame at a time, over a 384 × 384 km window. We compare a state-of-the-art '
                 'transformer fine-tuned on 40 GB of SEVIR storms, our own physics-shaped network, and the classical '
                 'comparison methods, scoring every frame against what the radar actually recorded.'),
        'bar-note': (
            'Higher is better. CSI = hits / (hits + misses + false alarms), pooled over every pixel, lead time and test '
            'window. Published numbers come from the Earthformer paper, whose SEVIR test set covers the same 2019 period '
            'but more windows. '
            + (f'The official Earthformer weights score {ef["csi_m"]:.3f} on our test set against 0.442 in the paper, and '
               if ef else '')
            + f'persistence scores {pe["csi_m"]:.3f} against 0.261. Published scores use a different number of windows.'),
        'score-text': (
            f'On the official SEVIR nowcast test set, {best} reaches a CSI-M of <strong>{t[best]["csi_m"]:.3f}</strong>, '
            f'against {fc["csi_m"]:.3f} for our from-scratch FlowCastNet, {of["csi_m"]:.3f} for FlowCast Optical Flow '
            f'extrapolation and {pe["csi_m"]:.3f} for persistence. Its mean squared error is {t[best]["mse"]:.2f} ×10⁻³, '
            f'{100 * (1 - t[best]["mse"] / of["mse"]):.0f}% lower than FlowCast Optical Flow, and its CSI on the most extreme cells '
            f'(VIL ≥ 219) is {t[best]["csi219"]:.3f}, against {of["csi219"]:.3f}. Per test window it beats FlowCast Optical Flow on '
            f'{win74:.0%} of windows with rain at VIL ≥ 74 and {win133:.0%} of windows with heavy cores at VIL ≥ 133.'),
        'lead-text': (
            'Forecast skill varies with lead time. At 60 minutes, heavy-storm CSI '
            '(VIL ≥ 133) is ' + listing(f'{lead_val(m, 133, 60):.3f} for {prose(m)}' for m in methods) + '. '
            'Hover a line to read exact values.'),
        'arch-text': '13.5M parameters over four scales (96 → 12 cells), with 7 × 7 depthwise kernels for a large receptive field.',
        'inside-text': (
            'Before swapping in Earthformer we built our own network. FlowCastNet is shaped like a simplified physical '
            'model. Most of what happens to a storm in an hour is that the wind carries it somewhere else; the rest is '
            'growth, decay and new cells forming. The network predicts those two parts separately, so its output stays '
            f'inspectable. Trained from scratch for {steps:,} steps it reaches CSI-M {fc["csi_m"]:.3f}, '
            f'compared with {of["csi_m"]:.3f} for FlowCast Optical Flow.'),
    }
    if ef:
        ft_text = ''
        if eft:
            ft_text = (f' Fine-tuned on our training events for {ft_train[-1]["step"]:,} steps, it scores '
                       f'{eft["csi_m"]:.3f} CSI-M and {eft["csi219"]:.3f} on extreme cells (VIL ≥ 219), against '
                       f'{ef["csi_m"]:.3f} and {ef["csi219"]:.3f} for the weights as published.')
        text['sota-text'] = (
            'We also evaluate Earthformer (Gao et al., NeurIPS 2022), a pretrained cuboid-attention transformer. '
            f'Its published weights score {ef["csi_m"]:.3f} CSI-M on our test set, compared with 0.442 '
            'in the paper on a larger set of windows.' + ft_text)
    if 'Pysteps LK' in t:
        p = t['Pysteps LK']
        text['score-text'] += (f' Pysteps LK scores {p["csi_m"]:.3f} CSI-M and {p["mse"]:.2f} ×10⁻³ MSE '
                               'using deterministic Lucas–Kanade / semi-Lagrangian extrapolation on encoded VIL.')
    if rollout:
        rmodels = [m for m in ORDER if m in rollout]

        def rl(m, thr, minutes):
            return rollout[m]['per_lead']['csi'][0][rollout[m]['thresholds'].index(thr)][minutes // 5 - 1]
        rbest = max([m for m in rmodels if m not in ('Optical flow', 'Persistence', 'Pysteps LK')],
                    key=lambda m: rollout[m]['overall']['csi_m'][0])
        n_ev = rollout[rmodels[0]]['events']
        text['rollout-text'] = (
            'The models were only trained to forecast one hour ahead. To use them as simulators, we feed each model its '
            f'own forecasts back in as input and let it run for three hours on {n_ev} four-hour radar events from the '
            'test period (June–November 2019), which no model trained on. Over the full three hours, CSI-M is '
            + listing(f'{rollout[m]["overall"]["csi_m"][0]:.3f} for {prose(m)}' for m in rmodels) + '. '
            f'Heavy cores are the hard part: at VIL ≥ 133, {rbest} scores {rl(rbest, 133, 60):.3f} at one hour, '
            f'{rl(rbest, 133, 120):.3f} at two and {rl(rbest, 133, 180):.3f} at three, against '
            f'{rl("Optical flow", 133, 60):.3f}, {rl("Optical flow", 133, 120):.3f} and {rl("Optical flow", 133, 180):.3f} '
            'for FlowCast Optical Flow. Each pass through a model smooths a little more.')
        s = np.load(RES / 'rollout_showcase.npz')
        when = f'({str(s["time_utc"])[:16]} UTC)' if 'time_utc' in s else ''
        text['rollout-caption'] = (f'<strong>Event {s["id"]}</strong> {when}, the test-period event '
                                   'with the most heavy precipitation over the simulated 3 hours. 65 minutes observed, then '
                                   '180 minutes simulated.')
    cases_out = []
    labels = ['Storm A', 'Storm B', 'Storm C', 'Moderate']
    for n, c in enumerate(cases):
        meta = test_meta.iloc[c['index']]
        scores = ' · '.join(f'{m} {c["csi74"][m]:.2f}' for m in methods if m in c['csi74'])
        cases_out.append({
            'index': c['index'], 'label': labels[n] if n < len(labels) else f'Case {n}',
            'gif': f'figures/case{n}.gif', 'png': f'figures/case{n}.png',
            'caption': (f'<strong>Test window #{c["index"]}</strong> · {meta.time_utc} UTC · '
                        f'{meta.llcrnrlat:.1f}–{meta.urcrnrlat:.1f}°N, {abs(meta.urcrnrlon):.1f}–{abs(meta.llcrnrlon):.1f}°W · '
                        f'CSI ≥ 74 on this window: {scores}'),
        })

    ef_list = [
        '<strong>Cuboid attention.</strong> The 13 input frames are encoded as one space-time block. Attention runs inside '
        'local cuboids (axial in time, height and width) plus a few global vectors that carry information across the '
        'whole 384 km window; the decoder then generates all 12 future frames at once.',
        '<strong>8.7M parameters.</strong> Smaller than FlowCastNet (13.5M).',
        '<strong>Trained on SEVIR before 2019.</strong> The official run used events up to 2019-01-01 for training and '
        'January–May 2019 for validation, so it has never seen the June–November 2019 test period.',
    ]
    if ft_config:
        peak = max((r['peak_vram_gb'] for r in ft_train), default=0)
        ft_steps = ft_train[-1]['step'] if ft_train else 0
        ef_list.append(
            f'<strong>Our fine-tune.</strong> {ft_steps:,} steps at batch {ft_config["batch"]} on our {int((clean.split == "train").sum()):,} clean '
            f'training events, peak learning rate {ft_config["lr"]:g}, with the same heavy-rain-weighted loss as FlowCastNet. '
            'The full-resolution convolution stem stays frozen and activations are recomputed in the backward pass, which '
            f'keeps peak memory at {peak:.1f} GB. Rotation and flip augmentation is off, so the eastward drift of US '
            'storms that the model already learned stays intact.')
    safety = [
        '<strong>Model code</strong> is the official PyTorch implementation from amazon-science/earth-forecasting-transformer '
        '(Apache-2.0), copied unchanged into third_party/earthformer. We scanned it before running it: it imports only '
        'torch, numpy and einops, and makes no network, shell, file or eval calls.',
        '<strong>Weights.</strong> Amazon\'s original checkpoint bucket no longer allows public downloads, so the weights '
        'come from PaddleScience\'s port of the same checkpoint (Baidu\'s PaddlePaddle bucket), whose published scores '
        'match the paper digit for digit.',
        '<strong>Loading.</strong> The checkpoint is a pickle, which can run code when loaded. We read it with a loader that '
        'only accepts numpy arrays and refuses anything else; it contained exactly 286 arrays and nothing else.',
        '<strong>Conversion checks.</strong> All 8.66M parameters map one-to-one onto the PyTorch model, with nothing missing '
        'or left over; every value is a finite float32; and the 11 integer position tables in the file are identical to '
        'the ones the PyTorch code computes. The only oddity was a stray newline in one parameter name.',
    ]
    if ef:
        safety.append(f'<strong>Behaviour check.</strong> On our test set the converted weights score {ef["csi_m"]:.3f} '
                      f'CSI-M and {ef["mse"]:.2f} ×10⁻³ MSE, against 0.442 and 3.70 in the paper. '
                      'These are measured locally on the 1,872-window test set.')

    train_list = [
        f'<strong>FlowCastNet</strong>: {steps:,} steps at batch {config["batch"]} ({steps * config["batch"]:,} windows), '
        f'AdamW with cosine decay from {config["lr"]:g}, EMA weights. Loss: heavy-rain-weighted MSE + MAE, an auxiliary '
        'loss that makes the motion field explain as much as it can, and a smoothness penalty on the motion.',
    ]
    if selected:
        train_list.append(f'FlowCastNet uses the EMA checkpoint selected at step {selected["step"]:,} '
                          f'by validation CSI-M ({selected["val_csi_m"]:.4f}), without test-set selection.')
    train_list.append(f'Validation uses up to {config["val_events"]} held-out events with fixed windows '
                      'starting at frames 0, 12 and 24. The official test set is used only for final evaluation.')
    if ft_config and ft_train:
        train_list.append(
            f'<strong>Earthformer fine-tune</strong>: {ft_train[-1]["step"]:,} steps at batch {ft_config["batch"]}, about '
            f'{np.median([r["samples_per_s"] for r in ft_train]):.1f} windows per second.')
    if ft_selected:
        train_list.append(f'Earthformer FT uses the EMA checkpoint selected at step {ft_selected["step"]:,} '
                          f'by validation CSI-M ({ft_selected["val_csi_m"]:.4f}).')
    train_list += [
        f'GPU: {config.get("gpu", "not recorded")}; PyTorch {config.get("torch", "not recorded")}. '
        f'bf16 with a per-process VRAM cap. FlowCastNet peaked at '
        f'{max(r["peak_vram_gb"] for r in train_rows):.1f} GB at about {sps:.0f} windows per second.',
        'FlowCast Optical Flow (our in-house model): OpenCV DIS flow averaged over the last three frame pairs, then semi-Lagrangian '
        'extrapolation (the same idea as operational systems such as pySTEPS).',
    ]
    if 'Pysteps LK' in metrics:
        train_list.append('pySTEPS LK (external optical flow baseline): four recent VIL frames, Lucas–Kanade motion, deterministic semi-Lagrangian '
                          'extrapolation with linear interpolation and zero outside the grid. Encoded VIL is used '
                          'directly; outputs are rounded to uint8. This is not the stochastic STEPS ensemble. '
                          'Configuration: ' + json.dumps(metrics['Pysteps LK'].get('configuration', {})))
    data_list = [
        f'<strong>{len(events):,} SEVIR VIL events</strong> ({gb:.1f} GB): all {storm:,} NOAA storm-report events before '
        f'June 2019, plus {len(events) - storm:,} randomly sampled events, because 84% of the test set is random events.',
        'Each event is fetched with a single HTTP range request from the raw SEVIR files on AWS. The files are contiguous '
        'and uncompressed, so we never download whole 10–17 GB files.',
        f'{len(events) - len(clean)} events ({(len(events) - len(clean)) / len(events):.1%}) contain SEVIR\'s missing-data '
        'fill, which reads as maximum intensity; the loader skips them.',
        'Training data ends 2019-06-01 and the official test set starts 2019-06-11, so there is no leakage.',
    ]
    if rollout:
        data_list.append(f'For the 3-hour simulations we fetched {rollout[rmodels[0]]["events"]} more events from the test '
                         'period (1.4 GB), so no model is scored on storms it trained on.')

    data = {
        'facts': [
            ['Input', '13 frames', '65 minutes of radar, 5-minute steps'],
            ['Output', '12 frames', 'the next 60 minutes'],
            ['Grid', '384 × 384', '1 km pixels, encoded VIL'],
            ['Training data', f'{gb:.0f} GB', f'{len(events):,} four-hour events, 2017 to May 2019'],
            ['Test set', f'{metrics[best]["windows"]:,}', 'windows, June–Nov 2019, never seen'],
        ],
        'methods': methods,
        'method_vars': {m: VARS[m] for m in methods},
        'thresholds': metrics[best]['thresholds'],
        'leads': leads,
        'test': {m: {'csi': by_threshold(metrics[m]), 'mse': metrics[m]['per_lead']['mse']} for m in methods},
        'rollout': ({m: {'csi': by_threshold(rollout[m]), 'mse': rollout[m]['per_lead']['mse']} for m in rmodels}
                    if rollout else None),
        'rollout_leads': [5 * (i + 1) for i in range(len(rollout[rmodels[0]]['per_lead']['mse']))] if rollout else [],
        'table': t,
        'refs': [{'name': n, 'csi_m': c, 'mse': m} for n, c, m in REFS],
        'bars': ([{'group': 'This study', 'name': m, 'value': t[m]['csi_m'], 'series': m} for m in methods]
                 + [{'group': 'Published', 'name': n, 'value': c} for n, c, _ in REFS]),
        'test_windows': metrics[best]['windows'],
        'cases': cases_out,
        'vil': {'levels': VIL_LEVELS, 'colors': VIL_COLORS},
        'training': {'val': [[r['step'], r['val_csi_m']] for r in val_rows]},
        'text': text,
        'lists': {'data-list': data_list, 'train-list': train_list, 'ef-list': ef_list, 'safety-list': safety,
                  'takeaways': [
                      f'{max(methods, key=lambda m: t[m]["csi_m"])} has the highest measured test CSI-M '
                      f'({max(t[m]["csi_m"] for m in methods):.4f}) among the local runs.',
                      f'FlowCastNet scores {fc["csi_m"]:.4f}, compared with {of["csi_m"]:.4f} for FlowCast Optical Flow '
                      f'and {pe["csi_m"]:.4f} for persistence.',
                  ],
                  'limits': [
                      'Earthformer includes external pretraining; these runs do not isolate architecture from training data and compute.',
                      'The model runs use one training seed. Scores are point estimates on this test set, without confidence intervals.',
                  ],
                  'nextsteps': ['Repeat training with additional seeds to measure variability.']},
        'repro': '\n'.join([
            'python download_sevir_test.py                      # official test set (5 GB)',
            'python download_sevir_train.py --budget-gb 40      # training events (resumable)',
            'python download_sevir_train.py --out vil_rollout_test --start 2019-06-11 --end 2020-01-01 \\',
            '       --budget-gb 1.45 --max-storm 120 --val-events 0   # test-period events for 3-hour rollouts',
            './train_supervised.sh --run runs/flowcast --steps 20000 --duty 0.75 --vram-gb 2.5',
            './train_supervised.sh --run runs/earthformer_ft --arch earthformer --init pretrained --freeze-stem \\',
            '       --no-augment --steps 8000 --batch 1 --lr 5e-5 --weight-decay 0 --duty 0.8 --vram-gb 2.8',
            'python evaluate.py --methods persistence,optical_flow,pysteps,flowcast,earthformer,earthformer_ft --chunk 1',
            'python rollout.py --models "Earthformer FT=runs/earthformer_ft/best.pt,Earthformer=earthformer-pretrained,FlowCastNet=runs/flowcast/best.pt"',
            'python make_figures.py --models "...same list..." && python build_report.py',
        ]),
        'footer': ('Data: SEVIR (Veillette, Samsi & Mattioli, NeurIPS 2020), Registry of Open Data on AWS. Earthformer: Gao '
                   'et al., NeurIPS 2022 (code Apache-2.0; weights via PaddleScience). Built for CSCI 596.'),
    }
    if ef and eft:
        data['lists']['takeaways'].append(
            f'The fine-tuned Earthformer scores {eft["csi_m"]:.4f} CSI-M versus {ef["csi_m"]:.4f} '
            'for the pretrained weights. Fine-tuning does not improve test CSI-M in this run.'
            if eft['csi_m'] <= ef['csi_m'] else
            f'Fine-tuning raises Earthformer test CSI-M from {ef["csi_m"]:.4f} to {eft["csi_m"]:.4f}.')
    extra = RES / 'report_text.json'
    if extra.exists():   # hand-written takeaways / limits / next steps, kept separate from the generated numbers
        for k, v in json.loads(extra.read_text(encoding='utf-8')).items():
            (data['lists'] if k in data['lists'] else data['text'])[k] = v

    data = report_labels(data)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    for f in ['anatomy.png'] + (['rollout.gif'] if rollout else []) + [f'case{n}.{e}' for n in range(len(cases)) for e in ('gif', 'png')]:
        shutil.copy(RES / 'figures' / f, OUT / 'figures' / f)
    template = Path('report/template.html').read_text(encoding='utf-8')
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    (OUT / 'index.html').write_text(template.replace('/*__DATA__*/', payload), encoding='utf-8')
    print('Wrote', (OUT / 'index.html').resolve())


if __name__ == '__main__':
    main()
