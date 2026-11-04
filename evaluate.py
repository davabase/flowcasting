"""Score forecasts on the official SEVIR nowcast test set (1,872 windows, Jun-Nov 2019).

Methods: persistence, optical flow (DIS + semi-Lagrangian extrapolation) and FlowCastNet.
Metrics follow the SEVIR/Earthformer convention (see flowcast/metrics.py). Each method's results are
cached in results/metrics.json, plus per-sample scores in results/per_sample_<method>.npz.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flowcast import baselines
from flowcast.data import IN_FRAMES, OUT_FRAMES, iter_test_blocks
from flowcast.metrics import Scores, summary_row

CHUNK = 4   # windows per GPU batch (inference + scoring stay well under the VRAM cap)
NAMES = {'persistence': 'Persistence', 'optical_flow': 'Optical flow', 'flowcast': 'FlowCastNet',
         'earthformer': 'Earthformer', 'earthformer_ft': 'Earthformer FT'}


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
    parser.add_argument('--methods', default='persistence,optical_flow,flowcast')
    parser.add_argument('--checkpoint', default='runs/flowcast/best.pt')
    parser.add_argument('--ft-checkpoint', default='runs/earthformer_ft/best.pt', help='Fine-tuned Earthformer')
    parser.add_argument('--out', default='results')
    parser.add_argument('--limit', type=int, default=None, help='Score only the first N test windows')
    parser.add_argument('--chunk', type=int, default=CHUNK, help='Windows per GPU batch')
    args = parser.parse_args()
    CHUNK = args.chunk
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / 'metrics.json'
    all_results = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    device = 'cuda'
    torch.cuda.set_per_process_memory_fraction(min(1.0, 2.5e9 / torch.cuda.mem_get_info()[1]))

    model = None
    for method in args.methods.split(','):
        if method in ('flowcast', 'earthformer', 'earthformer_ft'):
            from flowcast.predict import forecast, load_model
            source = {'flowcast': args.checkpoint, 'earthformer': 'earthformer-pretrained',
                      'earthformer_ft': args.ft_checkpoint}[method]
            model = load_model(source, device)
        scores = Scores(OUT_FRAMES, device=device)
        samples = {}
        started, count = time.time(), 0
        for start, block in iter_test_blocks():
            if args.limit is not None and start >= args.limit:
                break
            if args.limit is not None:
                block = block[:args.limit - start]
            inputs, target = block[:, :IN_FRAMES], block[:, IN_FRAMES:]
            fc = baselines.optical_flow(inputs)[0] if method == 'optical_flow' else None

            def chunk_forecasts():
                # One chunk at a time, so only CHUNK forecasts ever live on the GPU.
                for i in range(0, len(inputs), CHUNK):
                    if method == 'persistence':
                        yield torch.as_tensor(baselines.persistence(inputs[i:i + CHUNK]), device=device)
                    elif method == 'optical_flow':
                        yield torch.as_tensor(fc[i:i + CHUNK], device=device)
                    else:
                        yield forecast(model, inputs[i:i + CHUNK], batch=CHUNK)[0]['forecast']

            for i, pred in enumerate(chunk_forecasts()):
                tgt = torch.as_tensor(target[i * CHUNK:(i + 1) * CHUNK], device=device).float()
                pred = pred.float()
                scores.update(pred, tgt)
                for k, v in per_sample(pred, tgt).items():
                    samples.setdefault(k, []).append(v)
            count += len(block)
            print(f'{method}: {count} windows scored ({time.time() - started:.0f}s)', flush=True)
        result = scores.result()
        result['windows'] = count
        all_results[NAMES[method]] = result
        np.savez(out_dir / f'per_sample_{method}.npz', **{k: np.concatenate(v) for k, v in samples.items()})
        metrics_path.write_text(json.dumps(all_results, indent=1))
        print(json.dumps({k: round(v, 4) for k, v in summary_row(result).items()}), flush=True)

    print('\n' + ' | '.join(['Method'] + list(summary_row(next(iter(all_results.values()))).keys())))
    for name, r in all_results.items():
        print(' | '.join([name] + [f'{v:.4f}' for v in summary_row(r).values()]))


if __name__ == '__main__':
    main()
