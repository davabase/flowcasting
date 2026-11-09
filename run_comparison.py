"""Resume configured model runs, evaluate their best checkpoints, and build the report.

Progress is recorded in logs/comparison_status.json and logs/comparison_<stage>.log.
Safe to rerun after interruption: completed training schedules are skipped.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import torch

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs'


def status(state, stage, **extra):
    value = {'state': state, 'stage': stage, 'pid': os.getpid(),
             'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
    temporary = LOGS / 'comparison_status.json.tmp'
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(LOGS / 'comparison_status.json')


def execute(stage, arguments):
    status('running', stage, command=arguments)
    with (LOGS / f'comparison_{stage}.log').open('a', encoding='utf-8') as output:
        print(f'\n{datetime.now(timezone.utc).isoformat()} {arguments}', file=output, flush=True)
        subprocess.run([sys.executable, '-u', *arguments], cwd=ROOT, stdout=output,
                       stderr=subprocess.STDOUT, check=True)


def resume(name):
    run = ROOT / 'runs' / name
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    checkpoint = run / 'last.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        completed = saved['step'] >= config['steps']
        del saved
        if completed:
            return
    arguments = ['train.py']
    for key, value in config.items():
        if key in ('model', 'gpu', 'torch', 'resume'):
            continue
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                arguments.append(flag)
        else:
            arguments.extend([flag, str(value)])
    execute(name, [*arguments, '--resume'])


def main():
    LOGS.mkdir(exist_ok=True)
    # An OS-held lock is released automatically if this runner is interrupted.
    with (LOGS / 'comparison.lock').open('a+b') as lock:
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise SystemExit('A comparison runner is already active.')
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            resume('flowcast')
            resume('earthformer_ft')
            execute('evaluate', ['evaluate.py', '--methods', 'flowcast,earthformer,earthformer_ft', '--chunk', '1'])
            execute('figures', ['make_figures.py', '--models',
                               'Earthformer=earthformer-pretrained,Earthformer FT=runs/earthformer_ft/best.pt,'
                               'FlowCastNet=runs/flowcast/best.pt'])
            execute('report', ['build_report.py'])
            status('complete', 'report', report='results/report/index.html')
        except Exception as error:
            status('failed', 'pipeline', error=str(error))
            raise


if __name__ == '__main__':
    main()
