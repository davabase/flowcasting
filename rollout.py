"""Long-horizon simulation: roll models out autoregressively for 3 hours and score against radar.

Each model receives 65 minutes of observations from a 4-hour SEVIR event and keeps feeding its own
forecasts back in as context, producing 36 frames (180 minutes). Optical-flow extrapolation and
persistence are rolled out over the same horizon.

Use events no model has trained on. The default store holds raw events from the test period
(2019-06-11 onward), fetched with:
    python download_sevir_train.py --out vil_rollout_test --start 2019-06-11 --end 2020-01-01 \
        --budget-gb 1.45 --max-storm 120 --val-events 0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flowcast import baselines
from flowcast.data import IN_FRAMES, ROOT, load_manifest, read_event
from flowcast.metrics import Scores
from flowcast.predict import load_model, rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', default='FlowCastNet=runs/flowcast/best.pt',
                        help='Comma-separated name=checkpoint pairs ("earthformer-pretrained" for the official weights)')
    parser.add_argument('--events', default=str(ROOT / 'vil_rollout_test'), help='Event store to roll out on')
    parser.add_argument('--out', default='results')
    parser.add_argument('--steps', type=int, default=36)
    parser.add_argument('--batch', type=int, default=2, help='Windows per GPU forward pass')
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    torch.cuda.set_per_process_memory_fraction(min(1.0, 2.5e9 / torch.cuda.mem_get_info()[1]))
    models = {}
    for item in args.models.split(','):
        name, source = item.split('=', 1)
        models[name] = load_model(source, device)
    events = load_manifest(directory=args.events)
    names = list(models) + ['Optical flow', 'Persistence']
    scores = {n: Scores(args.steps, device=device) for n in names}
    showcase, best_cover = None, -1.0
    for i in range(0, len(events), 4):
        rows = events.iloc[i:i + 4]
        x = np.stack([read_event(s, args.events) for s in rows.slot])
        inputs, truth = x[:, :IN_FRAMES], x[:, IN_FRAMES:IN_FRAMES + args.steps]
        preds = {n: rollout(m, inputs, args.steps, batch=args.batch) for n, m in models.items()}
        preds['Optical flow'] = baselines.optical_flow(inputs, args.steps)[0]
        preds['Persistence'] = baselines.persistence(inputs, args.steps)
        target = torch.as_tensor(truth, device=device).float()
        for n in names:
            scores[n].update(torch.as_tensor(preds[n], device=device).float(), target)
        cover = (truth >= 133).mean(axis=(1, 2, 3))
        j = int(np.argmax(cover))
        if cover[j] > best_cover:
            best_cover = cover[j]
            showcase = {'slot': int(rows.slot.iloc[j]), 'id': rows.id.iloc[j], 'time_utc': rows.time_utc.iloc[j],
                        'inputs': inputs[j], 'truth': truth[j],
                        **{n: np.clip(np.rint(preds[n][j]), 0, 255).astype(np.uint8) for n in names}}
        print(f'{min(i + 4, len(events))}/{len(events)} events', flush=True)
    results = {n: {**s.result(), 'events': len(events), 'event_store': Path(args.events).name} for n, s in scores.items()}
    (out_dir / 'rollout_metrics.json').write_text(json.dumps(results, indent=1))
    np.savez_compressed(out_dir / 'rollout_showcase.npz', names=np.array(names),
                        **{k.replace(' ', '_'): v for k, v in showcase.items()})
    for n, r in results.items():
        k133 = r['thresholds'].index(133)
        csi = np.array(r['per_lead']['csi'][0][k133])
        print(f'{n:15s} CSI-133 @60/120/180 min: {csi[11]:.3f} {csi[23]:.3f} {csi[35]:.3f}   '
              f'CSI-M (3h) {r["overall"]["csi_m"][0]:.3f}')


if __name__ == '__main__':
    main()
