"""Train the generative texture refiner (flowcast/refiner.py) on top of a frozen forecaster.

Losses, per training window, with K noise samples from the generator:
  * adversarial: hinge loss from a conditional PatchGAN (last 4 observed frames + 12 forecast frames);
  * pool regularisation (NowcastNet-style): the mean over the K samples of the 4 km max-pooled forecast must match
    the max-pooled observation (intensity-weighted MSE + MAE). Averaging over samples, not pixels, lets every
    individual sample stay sharp while the ensemble stays faithful to the observations.
  * optional focal Tversky (--tversky) on the same pooled ensemble mean: misses of each VIL threshold region cost
    more than false alarms, which counters the refiner's tendency to shrink intense cores.
All terms are gradient-balanced on the generator output (adversarial pull = --adv x pool pull, etc.).

    python train_refiner.py --base runs/sharp/ef_original/last.pt --run runs/refine/ef
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from flowcast.data import IN_FRAMES, TrainWindows
from flowcast.losses import focal_tversky, wmse
from flowcast.predict import load_model
from flowcast.refiner import TextureDiscriminator, TextureGenerator


def pooled(samples, target, k=4):
    """Mean over samples of the k x k max-pooled forecast, and the max-pooled observation.
    samples (B, K, T, H, W), target (B, T, H, W) on [0, 1]."""
    b, n, t, h, w = samples.shape
    ps = F.max_pool2d(samples.reshape(b * n * t, 1, h, w), k).view(b, n, t, h // k, w // k).mean(1)
    pt = F.max_pool2d(target.reshape(b * t, 1, h, w), k).view(b, t, h // k, w // k)
    return ps, pt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True, help='Frozen forecaster checkpoint (any load_model source)')
    parser.add_argument('--run', required=True)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--samples', type=int, default=2, help='Noise samples per window (K)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d-lr', type=float, default=4e-4)
    parser.add_argument('--adv', type=float, default=0.3, help='Adversarial pull relative to pool regularisation')
    parser.add_argument('--tversky', type=float, default=0.0,
                        help='Focal Tversky pull on the pooled ensemble mean, relative to pool regularisation (0 = off)')
    parser.add_argument('--d-warmup', type=int, default=300, help='Discriminator-only steps before the generator trains')
    parser.add_argument('--d-steps', type=int, default=2, help='Discriminator updates per generator update')
    parser.add_argument('--warmup', type=int, default=300, help='Ramp the adversarial term in over this many steps')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--ckpt-every', type=int, default=500)
    parser.add_argument('--vram-gb', type=float, default=6.8)
    parser.add_argument('--duty', type=float, default=0.85)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_gb * 1e9 / torch.cuda.mem_get_info()[1]))
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)

    base = load_model(args.base, device).eval().requires_grad_(False)
    gen = TextureGenerator().to(device)
    disc = TextureDiscriminator().to(device)
    ema = TextureGenerator().to(device).requires_grad_(False)
    ema.load_state_dict(gen.state_dict())
    g_opt = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.0, 0.9))
    d_opt = torch.optim.Adam(disc.parameters(), lr=args.d_lr, betas=(0.0, 0.9))
    step, norms = 0, {}
    if args.resume and (run / 'last.pt').exists():
        ck = torch.load(run / 'last.pt', map_location=device, weights_only=False)
        gen.load_state_dict(ck['gen']); ema.load_state_dict(ck['ema']); disc.load_state_dict(ck['disc'])
        g_opt.load_state_dict(ck['g_opt']); d_opt.load_state_dict(ck['d_opt'])
        step, norms = ck['step'], ck.get('norms', {})
        print(f'Resumed from step {step}', flush=True)
    config = {'arch': 'refiner', 'base': args.base, 'samples': args.samples, 'adv': args.adv, 'd_steps': args.d_steps,
              'tversky': args.tversky}
    (run / 'config.json').write_text(json.dumps({**vars(args), 'model': config}, indent=1))
    print(f'generator {sum(p.numel() for p in gen.parameters()) / 1e6:.2f}M params, '
          f'discriminator {sum(p.numel() for p in disc.parameters()) / 1e6:.2f}M', flush=True)

    augment = getattr(base, 'config', {}).get('arch') != 'earthformer'   # Earthformer was trained without flips
    data = TrainWindows(samples_per_epoch=(args.steps - step) * args.batch, seed=10_000 + step, augment=augment)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, num_workers=args.workers, pin_memory=True,
                                         persistent_workers=True, prefetch_factor=4, drop_last=True)

    def save(name):
        torch.save({'gen': gen.state_dict(), 'ema': ema.state_dict(), 'disc': disc.state_dict(),
                    'g_opt': g_opt.state_dict(), 'd_opt': d_opt.state_dict(), 'step': step, 'norms': norms,
                    'config': config}, run / name)

    log = (run / 'log.jsonl').open('a')
    tic, running = time.time(), {}
    for x in loader:
        start = time.time()
        x = x.to(device, non_blocking=True).float().div_(255)
        inputs, target = x[:, :IN_FRAMES], x[:, IN_FRAMES:]
        context = inputs[:, -4:]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            fc = base(inputs.contiguous(memory_format=torch.channels_last))['forecast'].float()
        k = args.samples
        rep = lambda t: t.repeat_interleave(k, 0)          # noqa: E731
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = gen(rep(fc), rep(context)).float()        # (B*K, 12, H, W)

        # ---- generator: gradient-balanced pool regularisation + adversarial on the samples.
        # During the discriminator warm-up the generator stays at its initial identity (= the blurry base), so the
        # discriminator first learns what "too smooth" looks like before the generator starts chasing it.
        p = out.detach().requires_grad_(True)
        ps, pt = pooled(p.view(-1, k, *p.shape[1:]), target)
        l_pool = wmse(ps, pt)
        g_pool, = torch.autograd.grad(l_pool, p, retain_graph=args.tversky > 0)
        terms = [('pool', g_pool, 1.0)]
        if args.tversky > 0:
            l_tv = focal_tversky(ps, pt)
            g_tv, = torch.autograd.grad(l_tv, p)
            terms.append(('tv', g_tv, args.tversky))
        disc.requires_grad_(False)
        l_adv = -disc(rep(inputs), p).mean()
        g_adv, = torch.autograd.grad(l_adv, p)
        disc.requires_grad_(True)
        if step >= args.d_warmup:
            for name, g in [(t[0], t[1]) for t in terms] + [('adv', g_adv)]:
                n = float(g.norm()) + 1e-12
                norms[name] = n if name not in norms else 0.98 * norms[name] + 0.02 * n
            ramp = min(1.0, (step - args.d_warmup) / max(1, args.warmup))
            grad = g_pool + args.adv * ramp * norms['pool'] / norms['adv'] * g_adv
            for name, g, w in terms[1:]:
                grad = grad + w * norms['pool'] / norms[name] * g
            g_opt.zero_grad(set_to_none=True)
            out.backward(grad)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            g_opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()), list(gen.parameters()),
                                     1 - min(0.999, (1 + step) / (10 + step)))

        # ---- discriminator (two-timescale: several hinge steps per generator step, one sample each)
        fakes = out.detach().view(-1, k, *out.shape[1:])
        for j in range(args.d_steps):
            real = disc(inputs, target)
            fake = disc(inputs, fakes[:, j % k])
            l_d = F.relu(1 - real).mean() + F.relu(1 + fake).mean()
            d_opt.zero_grad(set_to_none=True)
            l_d.backward()
            d_opt.step()

        step += 1
        for name, v in (('pool', l_pool), ('adv', l_adv), ('d', l_d), ('d_real', real.mean()), ('d_fake', fake.mean()),
                        *((('tv', l_tv),) if args.tversky > 0 else ())):
            running[name] = running.get(name, 0.0) + float(v)
        if args.duty < 1.0:
            torch.cuda.synchronize()
            time.sleep(max(0.0, (time.time() - start) * (1 - args.duty) / args.duty))
        if step % args.log_every == 0:
            entry = {'step': step, 'windows_per_s': args.log_every * args.batch / (time.time() - tic),
                     'peak_vram_gb': torch.cuda.max_memory_allocated() / 1e9,
                     **{k2: v / args.log_every for k2, v in running.items()}}
            print(' '.join(f'{a}={b:.4g}' if isinstance(b, float) else f'{a}={b}' for a, b in entry.items()), flush=True)
            log.write(json.dumps(entry) + '\n')
            log.flush()
            tic, running = time.time(), {}
        if step % args.ckpt_every == 0 or step >= args.steps:
            save('last.pt')
        if step >= args.steps:
            break
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
