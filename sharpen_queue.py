"""Run the loss-function study: fine-tune, then score on the official test set, one experiment at a time.

Experiments are read from experiments/sharpen.txt on every iteration (so new ones can be appended while the
queue runs), one per line:  run_name | extra train.py args. Finished experiments are skipped: a run is done
once its name appears in results/sharp/metrics.json. Crashed training resumes from last.pt.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIST = ROOT / 'experiments' / 'sharpen.txt'
RESULTS = ROOT / 'results' / 'sharp' / 'metrics.json'
COMMON = {
    'earthformer': '--arch earthformer --init pretrained --checkpoint-level 0 --batch 1 --lr 5e-5 --warmup 200 '
                   '--weight-decay 0 --no-augment --workers 4 --steps 4000',
    'flowcast': '--arch flowcast --init-from runs/flowcast/best.pt --batch 4 --lr 1e-4 --warmup 200 --workers 6 '
                '--steps 5000',
}
SHARED = '--vram-gb 6.8 --duty 0.85 --log-every 100 --ckpt-every 500 --val-events 155'


def experiments():
    out = []
    for line in LIST.read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            name, _, extra = (s.strip() for s in line.partition('|'))
            out.append((name, extra))
    return out


def done(name):
    return RESULTS.exists() and name in json.loads(RESULTS.read_text())


def run(cmd, log):
    with open(log, 'a', encoding='utf-8') as f:
        f.write(f'\n$ {cmd}\n')
        f.flush()
        return subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                               env={**os.environ, 'PYTHONUTF8': '1'})


def main():
    (ROOT / 'logs').mkdir(exist_ok=True)
    base = ROOT / 'results' / 'sharp_base' / 'metrics.json'
    if not base.exists() or not {'Earthformer', 'FlowCastNet'} <= set(json.loads(base.read_text())):
        print(f'[{time.strftime("%H:%M:%S")}] scoring the original models (sharpness baseline)', flush=True)
        run(f'{sys.executable} evaluate.py --methods earthformer,flowcast --out results/sharp_base --vram-gb 5 '
            f'--duty 0.85', ROOT / 'logs' / 'sharp_base_eval.log')
    while True:
        pending = [(n, e) for n, e in experiments() if not done(n)]
        if not pending:
            print('queue empty', flush=True)
            return
        name, extra = pending[0]
        if extra.startswith('eval:'):        # evaluation-only entry, e.g. a scale blend of two trained models
            print(f'[{time.strftime("%H:%M:%S")}] {name}: evaluating {extra[5:]}', flush=True)
            code = run(f'{sys.executable} evaluate.py --methods "" --models "{name}={extra[5:]}" --out results/sharp '
                       f'--vram-gb 5 --duty 0.85', ROOT / 'logs' / f'sharp_eval_{name}.log')
            if code != 0 or not done(name):
                print(f'  evaluation failed ({code}); stopping queue', flush=True)
                return
            continue
        if extra.startswith('refine:'):      # generative texture refiner on a frozen base (train_refiner.py)
            run_dir = ROOT / 'runs' / 'refine' / name
            steps = int(re.findall(r'--steps (\d+)', extra)[-1])
            print(f'[{time.strftime("%H:%M:%S")}] {name}: training refiner', flush=True)
            for attempt in range(6):
                last = run_dir / 'last.pt'
                if last.exists():
                    import torch
                    if torch.load(last, map_location='cpu', weights_only=False)['step'] >= steps:
                        break
                code = run(f'{sys.executable} train_refiner.py {extra[7:]} --run {run_dir.as_posix()} '
                           f'{"--resume" if last.exists() else ""}', ROOT / 'logs' / f'sharp_{name}.log')
                if code == 0:
                    break
                print(f'  refiner exited with {code}; retrying ({attempt + 1})', flush=True)
                time.sleep(30)
            print(f'[{time.strftime("%H:%M:%S")}] {name}: evaluating', flush=True)
            code = run(f'{sys.executable} evaluate.py --methods "" --models "{name}={(run_dir / "last.pt").as_posix()}" '
                       f'--out results/sharp --vram-gb 5 --duty 0.85', ROOT / 'logs' / f'sharp_eval_{name}.log')
            if code != 0 or not done(name):
                print(f'  evaluation failed ({code}); stopping queue', flush=True)
                return
            continue
        arch ='flowcast' if '--arch flowcast' in extra or name.startswith('fc_') else 'earthformer'
        extra = extra.replace('--arch flowcast', '').replace('--arch earthformer', '')
        run_dir = ROOT / 'runs' / 'sharp' / name
        log = ROOT / 'logs' / f'sharp_{name}.log'
        steps = int(re.findall(r'--steps (\d+)', f'{COMMON[arch]} {extra}')[-1])   # argparse: last wins
        print(f'[{time.strftime("%H:%M:%S")}] {name}: training', flush=True)
        for attempt in range(6):
            last = run_dir / 'last.pt'
            if last.exists():
                import torch
                if torch.load(last, map_location='cpu', weights_only=False)['step'] >= steps:
                    break
            code = run(f'{sys.executable} train.py {COMMON[arch]} {SHARED} {extra} --run {run_dir.as_posix()} '
                       f'{"--resume" if last.exists() else ""}', log)
            if code == 0:
                break
            print(f'  training exited with {code}; retrying ({attempt + 1})', flush=True)
            time.sleep(30)
        print(f'[{time.strftime("%H:%M:%S")}] {name}: evaluating', flush=True)
        code = run(f'{sys.executable} evaluate.py --methods "" --models "{name}={(run_dir / "last.pt").as_posix()}" '
                   f'--out results/sharp --vram-gb 5 --duty 0.85', ROOT / 'logs' / f'sharp_eval_{name}.log')
        if code != 0 or not done(name):
            print(f'  evaluation failed ({code}); stopping queue', flush=True)
            return


if __name__ == '__main__':
    main()
