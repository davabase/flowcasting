"""Score forecasts on the official SEVIR nowcast test set (1,872 windows, Jun-Nov 2019).

Methods: persistence, DIS optical flow, pysteps Lucas-Kanade extrapolation and neural models, plus any
checkpoint via --models "Display name=path,...". Metrics follow the SEVIR/Earthformer convention (see
flowcast/metrics.py), plus sharpness metrics (flowcast/sharpness.py: power spectrum, FSS, intensity histogram).
Each method's results are cached in <out>/metrics.json, plus per-sample scores in <out>/per_sample_<key>.npz.
"""
import argparse
import json
import os
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # before torch
from pathlib import Path

import numpy as np
import torch

from flowcast import baselines
from flowcast.optical_flow import optical_flow
from flowcast.pysteps import pysteps
from flowcast.data import IN_FRAMES, OUT_FRAMES, iter_test_blocks
from flowcast.metrics import Scores, summary_row
from flowcast.sharpness import Sharpness, sharp_row

CHUNK = 4   # windows per GPU batch (inference + scoring stay well under the VRAM cap)
NAMES = {'persistence': 'Persistence', 'optical_flow': 'Optical flow', 'flowcast': 'FlowCastNet',
         'earthformer': 'Earthformer', 'earthformer_ft': 'Earthformer FT', 'pysteps': 'Pysteps LK'}


def per_sample(pred, target):
    """Per-sample CSI at 74/133 (all leads pooled) and MSE, for picking showcase cases."""
    out = {}
    for t in (74, 133):
        a, b = pred >= t, target >= t
        h = (a & b).sum(dim=(1, 2, 3)).float()
        out[f'csi{t}'] = (h / ((a | b).sum(dim=(1, 2, 3)).float().clamp(min=1))).cpu().numpy()
        out[f'cover{t}'] = b.float().mean(dim=(1, 2, 3)).cpu().numpy()
    out['mse'] = (((pred - target) / 255) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    return out


def main():
    global CHUNK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', default='persistence,optical_flow,pysteps,flowcast')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=4, help='PyTorch CPU threads')
    parser.add_argument('--checkpoint', default='runs/flowcast/best.pt')
    parser.add_argument('--ft-checkpoint', default='runs/earthformer_ft/best.pt', help='Fine-tuned Earthformer')
    parser.add_argument('--out', default='results')
    parser.add_argument('--limit', type=int, default=None, help='Score only the first N test windows')
    parser.add_argument('--chunk', type=int, default=CHUNK, help='Windows per GPU batch')
    parser.add_argument('--models', default='', help='Extra checkpoints: "Display name=path,Other=path2"')
    parser.add_argument('--vram-gb', type=float, default=2.5)
    parser.add_argument('--duty', type=float, default=1.0, help='Fraction of wall time on the GPU (<1 throttles)')
    args = parser.parse_args()
    CHUNK = args.chunk
    methods = [m for m in args.methods.split(',') if m]     # --methods "" scores only --models
    if any(m not in NAMES for m in methods):
        parser.error(f'Methods must be chosen from {list(NAMES)}')
    if args.chunk < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error('--chunk, --threads and --limit must be positive')
    torch.set_num_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / 'metrics.json'
    all_results = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    device = args.device
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_gb * 1e9 / torch.cuda.mem_get_info()[1]))

    jobs = []   # (key, display name, model source or None)
    for method in methods:
        source = {'flowcast': args.checkpoint, 'earthformer': 'earthformer-pretrained',
                  'earthformer_ft': args.ft_checkpoint}.get(method)
        jobs.append((method, NAMES[method], source))
    for item in filter(None, args.models.split(',')):
        name, _, source = item.partition('=')
        jobs.append((''.join(c if c.isalnum() else '_' for c in name.lower()), name, source))

    model = None
    for method, display, source in jobs:
        if source is not None:
            from flowcast.predict import forecast, load_model
            model = load_model(source, device)
        # Earthformer's full-resolution conv stem needs ~1 GB per window at inference: one window per batch.
        is_earthformer = source is not None and getattr(model, 'config', {}).get('arch') == 'earthformer'
        CHUNK = 1 if is_earthformer else args.chunk
        scores = Scores(OUT_FRAMES, device=device)
        sharp = Sharpness(OUT_FRAMES, device=device)
        samples = {}
        started, count = time.time(), 0
        for start, block in iter_test_blocks():
            if args.limit is not None and start >= args.limit:
                break
            if args.limit is not None:
                block = block[:args.limit - start]
            inputs, target = block[:, :IN_FRAMES], block[:, IN_FRAMES:]
            fc = optical_flow(inputs)[0] if method == 'optical_flow' else None
            if method == 'pysteps':
                fc = pysteps(inputs)[0]

            def chunk_forecasts():
                # One chunk at a time, so only CHUNK forecasts ever live on the GPU.
                for i in range(0, len(inputs), CHUNK):
                    if method == 'persistence':
                        yield torch.as_tensor(baselines.persistence(inputs[i:i + CHUNK]), device=device)
                    elif method in ('optical_flow', 'pysteps'):
                        yield torch.as_tensor(fc[i:i + CHUNK], device=device)
                    else:
                        yield forecast(model, inputs[i:i + CHUNK], batch=CHUNK)[0]['forecast']

            tick = time.time()
            for i, pred in enumerate(chunk_forecasts()):
                tgt = torch.as_tensor(target[i * CHUNK:(i + 1) * CHUNK], device=device).float()
                pred = pred.float()
                scores.update(pred, tgt)
                sharp.update(pred, tgt)
                for k, v in per_sample(pred, tgt).items():
                    samples.setdefault(k, []).append(v)
                if args.duty < 1.0:
                    if device.startswith('cuda'):
                        torch.cuda.synchronize()
                    time.sleep(max(0.0, (time.time() - tick) * (1 - args.duty) / args.duty))
                    tick = time.time()
            count += len(block)
            print(f'{method}: {count} windows scored ({time.time() - started:.0f}s)', flush=True)
        result = scores.result()
        result['sharp'] = sharp.result()
        result['windows'] = count
        result['elapsed_seconds'] = time.time() - started
        result['source'] = source
        if method == 'pysteps':
            from importlib.metadata import version
            result['configuration'] = {
                'package': 'pysteps', 'version': version('pysteps'), 'motion': 'LK',
                'motion_frames': 4, 'extrapolation': 'semilagrangian',
                'interp_order': 1, 'n_iter': 1, 'outval': 0,
                'input_scale': 'encoded SEVIR VIL 0-255', 'output': 'rounded uint8',
            }
        all_results[display] = result
        np.savez(out_dir / f'per_sample_{method}.npz', **{k: np.concatenate(v) for k, v in samples.items()})
        metrics_path.write_text(json.dumps(all_results, indent=1))
        print(json.dumps({k: round(v, 4) for k, v in {**summary_row(result), **sharp_row(result)}.items()}),
              flush=True)

    def row(r):
        return {**summary_row(r), **(sharp_row(r) if 'sharp' in r else {})}
    first = next(iter(all_results.values()))
    print('\n' + ' | '.join(['Method'] + list(row(first).keys())))
    for name, r in all_results.items():
        print(' | '.join([name] + [f'{v:.4f}' for v in row(r).values()]))


if __name__ == '__main__':
    main()
