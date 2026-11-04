"""Train FlowCastNet on SEVIR VIL: 13 past radar frames (65 min) -> next 12 frames (60 min).

GPU-friendly by design: bf16 autocast, a hard per-process VRAM cap (fails instead of spilling into
shared memory on Windows), and an optional duty cycle that leaves compute for other GPU users.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from flowcast.data import IN_FRAMES, OUT_FRAMES, TrainWindows, ValWindows
from flowcast.metrics import Scores
from flowcast.model import FlowCastNet, warp


def weighted_loss(pred, target):
    """Balanced MSE + MAE (Shi et al. 2017 style): intense echoes weigh up to 8x, so the model is not
    rewarded for forecasting a uniform drizzle. Inputs are on the [0, 1] scale."""
    v = target * 255
    weight = 1 + (v >= 16).float() + 2 * (v >= 74).float() + 4 * (v >= 133).float()
    err = pred - target
    return (weight * (err * err + err.abs())).mean()


def _blur(z):
    return F.avg_pool2d(z, 9, stride=1, padding=4, count_include_pad=False)


def advection_loss(inputs, out, target):
    """Make the motion branch explain as much as possible on its own (NowcastNet-style evolution loss).

    Bilinear warping only has local gradients, so we also warp a blurred copy of the last frame and
    compare it with the blurred target: that widens the basin for learning large displacements.
    """
    coarse = warp(_blur(inputs[:, -1:].float()), out['flow'])
    return weighted_loss(out['advected'], target) + 2 * (coarse - _blur(target)).abs().mean()


def smoothness(flow_latent):
    """Total variation of the displacement fields (latent grid) keeps the learned motion physical."""
    return ((flow_latent[..., :, 1:] - flow_latent[..., :, :-1]).abs().mean()
            + (flow_latent[..., 1:, :] - flow_latent[..., :-1, :]).abs().mean())


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    scores = Scores(OUT_FRAMES, pools=(1,), device=device)
    for x in loader:
        x = x.to(device, non_blocking=True).float()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model((x[:, :IN_FRAMES] / 255).contiguous(memory_format=torch.channels_last))
        scores.update(out['forecast'] * 255, x[:, IN_FRAMES:])
    r = scores.result()
    model.train()
    k = r['thresholds'].index(133)
    return {'val_csi_m': r['overall']['csi_m'][0], 'val_csi_133': r['overall']['csi'][0][k],
            'val_bias_133': r['overall']['bias'][0][k], 'val_mse': r['overall']['mse']}


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/flowcast')
    parser.add_argument('--steps', type=int, default=60000)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--warmup', type=int, default=1500)
    parser.add_argument('--weight-decay', type=float, default=0.02)
    parser.add_argument('--smooth', type=float, default=0.01, help='Flow smoothness weight')
    parser.add_argument('--adv', type=float, default=0.5, help='Auxiliary advection-only loss weight')
    parser.add_argument('--ema', type=float, default=0.999)
    parser.add_argument('--arch', choices=['flowcast', 'earthformer'], default='flowcast')
    parser.add_argument('--init', choices=['scratch', 'pretrained'], default='scratch',
                        help='earthformer only: start from the official SEVIR weights')
    parser.add_argument('--freeze-stem', action='store_true', help='earthformer only: freeze the full-resolution conv stem')
    parser.add_argument('--checkpoint-level', type=int, default=2,
                        help='earthformer only: activation checkpointing (0 = off, 2 = most memory saved)')
    parser.add_argument('--widths', default='96,192,384,384')
    parser.add_argument('--depths', default='2,2,4,2')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--val-every', type=int, default=5000)
    parser.add_argument('--val-events', type=int, default=60)
    parser.add_argument('--log-every', type=int, default=200)
    parser.add_argument('--ckpt-every', type=int, default=1000, help='Save last.pt this often (for --resume)')
    parser.add_argument('--vram-gb', type=float, default=3.0, help='Hard cap on this process\'s VRAM')
    parser.add_argument('--duty', type=float, default=1.0, help='Fraction of wall time spent computing (<1 throttles)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-augment', action='store_true', help='Disable rotation/flip augmentation')
    args = parser.parse_args()

    device = 'cuda'
    torch.backends.cudnn.benchmark = False     # autotuning grabs large workspaces; not worth it here
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_gb * 1e9 / torch.cuda.mem_get_info()[1]))
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)

    if args.arch == 'earthformer':
        from flowcast.earthformer import EarthformerNowcaster, load_pretrained
        if args.init == 'pretrained':
            model = load_pretrained(device, checkpoint_level=args.checkpoint_level)[0].train()
        else:
            model = EarthformerNowcaster(args.checkpoint_level).to(device)
        model.config = {'arch': 'earthformer'}     # inference rebuilds it without activation checkpointing
        if args.freeze_stem:   # frozen full-resolution conv stem stores no activations for backward (3.4 -> 2.2 GB)
            model.net.initial_encoder.requires_grad_(False)
    else:
        model = FlowCastNet(widths=[int(w) for w in args.widths.split(',')],
                            depths=[int(d) for d in args.depths.split(',')]).to(device)
    model = model.to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).requires_grad_(False)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 or name.endswith('gamma') else decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=args.lr, betas=(0.9, 0.99))
    step, best = 0, -1.0
    if args.resume and (run / 'last.pt').exists():
        ckpt = torch.load(run / 'last.pt', map_location=device)
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['opt'])
        step, best = ckpt['step'], ckpt.get('best', -1.0)
        print(f'Resumed from step {step}', flush=True)

    train = TrainWindows(samples_per_epoch=(args.steps - step) * args.batch, seed=step, augment=not args.no_augment)
    val = ValWindows(max_events=args.val_events)
    print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params | '
          f'{len(train.slots)} training events | {len(val)} validation windows', flush=True)
    (run / 'config.json').write_text(json.dumps({**vars(args), 'model': model.config}, indent=1))
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch, num_workers=args.workers,
                                         pin_memory=True, persistent_workers=True, prefetch_factor=4,
                                         drop_last=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=8 if args.arch == 'flowcast' else args.batch, num_workers=2)
    log = (run / 'log.jsonl').open('a')
    ema_params, model_params = list(ema.parameters()), list(model.parameters())

    def checkpoint(name, metrics=None):
        torch.save({'model': model.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(),
                    'step': step, 'best': best, 'config': model.config, 'metrics': metrics}, run / name)

    model.train()
    tic, seen, running = time.time(), 0, {}
    for x in loader:
        start = time.time()
        x = x.to(device, non_blocking=True).float().div_(255)
        inputs = x[:, :IN_FRAMES].contiguous(memory_format=torch.channels_last)
        target = x[:, IN_FRAMES:]
        for group in opt.param_groups:
            group['lr'] = lr_at(step, args)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(inputs)
        data_loss = weighted_loss(out['forecast'], target)
        if 'flow_latent' in out:          # FlowCastNet's motion-field terms
            adv_loss = advection_loss(inputs, out, target)
            smooth_loss = smoothness(out['flow_latent'])
        else:
            adv_loss = smooth_loss = torch.zeros((), device=device)
        loss = data_loss + args.adv * adv_loss + args.smooth * smooth_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model_params, 1.0)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, model_params, 1 - min(args.ema, (1 + step) / (10 + step)))
        step += 1
        seen += x.shape[0]
        for k, v in (('loss', data_loss), ('adv', adv_loss), ('smooth', smooth_loss), ('grad', grad_norm),
                     ('flow_px', out['flow_latent'][:, -1].norm(dim=1).mean() if 'flow_latent' in out else 0.0)):
            running[k] = running.get(k, 0.0) + float(v)
        if args.duty < 1.0:
            torch.cuda.synchronize()
            time.sleep((time.time() - start) * (1 - args.duty) / args.duty)

        if step % args.log_every == 0:
            elapsed = time.time() - tic
            entry = {'step': step, 'lr': lr_at(step, args), 'samples_per_s': seen / elapsed,
                     'peak_vram_gb': torch.cuda.max_memory_allocated() / 1e9,
                     **{k: v / args.log_every for k, v in running.items()}}
            print(' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in entry.items()), flush=True)
            log.write(json.dumps(entry) + '\n')
            log.flush()
            tic, seen, running = time.time(), 0, {}
        if step % args.val_every == 0 or step == args.steps:
            metrics = validate(ema, val_loader, device)
            print(f'[val @ {step}] ' + ' '.join(f'{k}={v:.4f}' for k, v in metrics.items()), flush=True)
            log.write(json.dumps({'step': step, **metrics}) + '\n')
            log.flush()
            if metrics['val_csi_m'] > best:
                best = metrics['val_csi_m']
                checkpoint('best.pt', metrics)
            checkpoint('last.pt', metrics)
            tic = time.time()
        elif step % args.ckpt_every == 0:
            checkpoint('last.pt')
        if step >= args.steps:
            break
    print(f'Done. Best validation CSI-M {best:.4f}', flush=True)


if __name__ == '__main__':
    main()
