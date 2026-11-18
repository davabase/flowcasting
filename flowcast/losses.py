"""Loss functions borrowed from medical image segmentation and reconstruction, adapted to VIL nowcasting.

Pixel losses (MSE) make deterministic forecasts converge to the average of the plausible futures,
which is smooth. Medical imaging has the same tension between pixel-accurate and anatomically crisp
outputs, and several families of losses were built to fight it:

  region overlap   soft Jaccard / Dice (Milletari et al. 2016, V-Net; Rahman & Wang 2016). The Jaccard
                   index of a thresholded field *is* the CSI used to score nowcasts.
                   Focal Tversky (Salehi et al. 2017; Abraham & Khan 2019): misses cost more than false
                   alarms, which targets the under-forecast intense cores.
  boundary         Boundary loss (Kervadec et al., MIDL 2019): mass placed far from the true region's
                   boundary is penalised in proportion to its signed distance.
                   Hausdorff-distance loss (Karimi & Salcudean, IEEE TMI 2019): pixel errors weighted by
                   their distance to the true and predicted boundaries.
  structure        multi-scale SSIM (Zhao et al. 2017, image restoration losses).
  frequency        k-space / power-spectrum matching, as in MRI reconstruction; the spectrum ignores phase,
                   so it rewards realistic detail without the double penalty of pixel losses.
                   Gradient difference loss (Mathieu et al. 2016; used in CT/MR synthesis).
  adversarial      PatchGAN discriminator (pix2pix; MR-to-CT synthesis), hinge loss + spectral norm.

Segmentation losses act on soft threshold masks sigmoid((VIL - t) / tau) at the six SEVIR thresholds.

`LossMixer` combines the terms by *gradient balancing* on the model output: each auxiliary term is
rescaled so its gradient norm is `weight` x the anchor term's (running averages), so a weight of 0.5
means "half as much pull as the pixel loss" regardless of the loss's native scale.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

THRESHOLDS = (16, 74, 133, 160, 181, 219)
TAU = 5.0          # VIL units: width of the soft threshold
RADIUS = 24        # px (= km): distance maps are capped here
DIST_THRESHOLDS = (74, 133, 181)   # distance-map losses use 3 thresholds (cost)


# ----------------------------------------------------------------------------- masks and distance maps
def _thr(x, thresholds):
    return torch.tensor(thresholds, device=x.device, dtype=x.dtype).view(-1, 1, 1, 1, 1)


def soft_masks(pred, thresholds=THRESHOLDS, tau=TAU):
    """pred (B, T, H, W) on [0, 1] -> (K, B, T, H, W) soft exceedance probabilities."""
    return torch.sigmoid((pred.unsqueeze(0) * 255 - _thr(pred, thresholds)) / tau)


def hard_masks(target, thresholds=THRESHOLDS):
    return (target.unsqueeze(0) * 255 >= _thr(target, thresholds)).float()


@torch.no_grad()
def distance_to(mask, radius=RADIUS):
    """Chessboard distance (px, capped at radius + 1) from every pixel to the nearest pixel of `mask`.

    mask (..., H, W) in {0, 1}. Repeated 3x3 dilation on the GPU: cheap and exact for chessboard distance.
    """
    shape = mask.shape
    cur = mask.reshape(-1, 1, *shape[-2:]).float()
    dist = 1 - cur
    for _ in range(radius):
        cur = F.max_pool2d(cur, 3, 1, 1)
        dist = dist + (1 - cur)
    return dist.reshape(shape)


@torch.no_grad()
def signed_distance(mask, radius=RADIUS):
    """Positive outside the region (distance to it), negative inside (distance to the background), / (R+1)."""
    return (distance_to(mask, radius) - distance_to(1 - mask, radius)) / (radius + 1)


# ----------------------------------------------------------------------------- individual losses
def mse(pred, target):
    return F.mse_loss(pred, target)


def _lowpass(z, sigma=3.0):
    """Separable Gaussian blur of (B, T, H, W): amplitude response 0.5 at 16 km, 0.06 at 8 km wavelengths."""
    radius = int(3 * sigma)
    x = torch.arange(-radius, radius + 1, device=z.device, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    flat = z.flatten(0, 1).unsqueeze(1).float()
    flat = F.conv2d(F.pad(flat, (radius, radius, 0, 0), mode='reflect'), k.view(1, 1, 1, -1))
    flat = F.conv2d(F.pad(flat, (0, 0, radius, radius), mode='reflect'), k.view(1, 1, -1, 1))
    return flat.view_as(z)


def lpmse(pred, target):
    """Scale-split anchor: MSE only on scales above ~16 km, where storms are predictable. Smaller scales are left
    to distribution-matching terms (spectral, gan), so they are not averaged away by the double penalty."""
    return F.mse_loss(_lowpass(pred), _lowpass(target))


def lpwmse(pred, target):
    """Scale-split version of FlowCastNet's intensity-weighted loss: balanced MSE + MAE above ~16 km only."""
    return wmse(_lowpass(pred), _lowpass(target))


def wmse(pred, target):
    """Balanced MSE + MAE (FlowCastNet's original loss): intense echoes weigh up to 8x."""
    v = target * 255
    weight = 1 + (v >= 16).float() + 2 * (v >= 74).float() + 4 * (v >= 133).float()
    err = pred - target
    return (weight * (err * err + err.abs())).mean()


def jaccard(pred, target):
    """Soft Jaccard (IoU) per threshold, summed over the whole batch like CSI is summed over a test set."""
    p, y = soft_masks(pred), hard_masks(target)
    dims = tuple(range(1, p.ndim))
    inter = (p * y).sum(dims)
    union = p.sum(dims) + y.sum(dims) - inter
    return (1 - (inter + 1) / (union + 1)).mean()


def dice(pred, target):
    p, y = soft_masks(pred), hard_masks(target)
    dims = tuple(range(1, p.ndim))
    return (1 - (2 * (p * y).sum(dims) + 1) / (p.sum(dims) + y.sum(dims) + 1)).mean()


def focal_tversky(pred, target, alpha=0.3, beta=0.7, gamma=0.75):
    """Tversky index with misses (beta) weighted above false alarms (alpha); focal exponent per Abraham & Khan."""
    p, y = soft_masks(pred), hard_masks(target)
    dims = tuple(range(1, p.ndim))
    tp = (p * y).sum(dims)
    fp = (p * (1 - y)).sum(dims)
    fn = ((1 - p) * y).sum(dims)
    ti = (tp + 1) / (tp + alpha * fp + beta * fn + 1)
    return ((1 - ti).clamp(min=1e-6) ** gamma).mean()


def boundary(pred, target):
    """Kervadec et al.: integral of the soft mask against the true region's signed distance map."""
    p, y = soft_masks(pred, DIST_THRESHOLDS), hard_masks(target, DIST_THRESHOLDS)
    return (signed_distance(y) * p).mean()


def hausdorff(pred, target, alpha=2.0):
    """Karimi & Salcudean HD-DT loss: squared mask error x (d_true^a + d_pred^a), distances to each boundary."""
    p, y = soft_masks(pred, DIST_THRESHOLDS), hard_masks(target, DIST_THRESHOLDS)
    with torch.no_grad():
        d_true = signed_distance(y).abs()
        d_pred = signed_distance((p > 0.5).float()).abs()
        weight = d_true ** alpha + d_pred ** alpha
    return ((p - y) ** 2 * weight).mean() * 10   # x10 only to keep logged values readable


def _gaussian(size=11, sigma=1.5, device='cpu'):
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def _ssim(x, y, window):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x, mu_y = F.conv2d(x, window), F.conv2d(y, window)
    sxx = F.conv2d(x * x, window) - mu_x ** 2
    syy = F.conv2d(y * y, window) - mu_y ** 2
    sxy = F.conv2d(x * y, window) - mu_x * mu_y
    return (((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))).mean()


def ssim(pred, target, scales=3):
    """1 - SSIM averaged over 3 scales (1, 2, 4 km)."""
    window = _gaussian(device=pred.device)
    x, y = pred.flatten(0, 1).unsqueeze(1), target.flatten(0, 1).unsqueeze(1)
    total = 0.0
    for s in range(scales):
        total = total + _ssim(x, y, window)
        if s < scales - 1:
            x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
    return 1 - total / scales


_RADIAL = {}


def _radial_index(h, w, device):
    key = (h, w, str(device))
    if key not in _RADIAL:
        ky = torch.fft.fftfreq(h, device=device) * h
        kx = torch.fft.rfftfreq(w, device=device) * w
        k = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).round().long().clamp(max=min(h, w) // 2)
        counts = torch.bincount(k.flatten(), minlength=min(h, w) // 2 + 1).float()
        win = torch.outer(torch.hann_window(h, periodic=False, device=device),
                          torch.hann_window(w, periodic=False, device=device))
        _RADIAL[key] = (k.flatten(), counts, win)
    return _RADIAL[key]


def radial_power(x):
    """x (N, H, W) float -> (N, K+1) radially averaged power spectrum (Hann window, orthonormal FFT).
    Bin k is the wavelength H / k pixels (km)."""
    n, h, w = x.shape
    index, counts, win = _radial_index(h, w, x.device)
    f = torch.fft.rfft2(x.float() * win, norm='ortho')
    power = (f.real ** 2 + f.imag ** 2).flatten(1)
    out = torch.zeros(n, counts.numel(), device=x.device).index_add_(1, index, power)
    return out / counts


def spectral(pred, target, eps=1e-6):
    """k-space loss: L1 between log radial power spectra, per frame, at every scale from 384 to 2 km."""
    pp = radial_power(pred.flatten(0, 1))[:, 1:]
    pt = radial_power(target.flatten(0, 1))[:, 1:]
    return (torch.log(pp + eps) - torch.log(pt + eps)).abs().mean()


def kspace(pred, target, eps=1e-6):
    """Full 2-D k-space loss (MRI-reconstruction style): L1 between log power at every spatial frequency and
    orientation. Unlike the radial version it cannot be satisfied by axis-aligned stripes, because stripes put
    their power on the kx = 0 / ky = 0 lines where real radar has little. Phase is ignored, so there is still
    no double penalty for misplaced small-scale detail."""
    _, _, win = _radial_index(*pred.shape[-2:], pred.device)
    fp = torch.fft.rfft2(pred.flatten(0, 1).float() * win, norm='ortho')
    ft = torch.fft.rfft2(target.flatten(0, 1).float() * win, norm='ortho')
    lp = torch.log(fp.real ** 2 + fp.imag ** 2 + eps)
    lt = torch.log(ft.real ** 2 + ft.imag ** 2 + eps)
    return (lp - lt).abs().mean()


def gdl(pred, target):
    """Gradient difference loss: match the magnitude of spatial gradients (edge strength)."""
    def dx(z):
        return z[..., :, 1:] - z[..., :, :-1]

    def dy(z):
        return z[..., 1:, :] - z[..., :-1, :]
    return ((dx(pred).abs() - dx(target).abs()).abs().mean() + (dy(pred).abs() - dy(target).abs()).abs().mean())


LOSSES = {'mse': mse, 'lpmse': lpmse, 'lpwmse': lpwmse, 'wmse': wmse, 'jaccard': jaccard, 'dice': dice, 'ftversky': focal_tversky,
          'boundary': boundary, 'hausdorff': hausdorff, 'ssim': ssim, 'spectral': spectral, 'kspace': kspace, 'gdl': gdl}


# ----------------------------------------------------------------------------- adversarial
class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN (pix2pix): last 4 observed frames + 12 forecast frames -> 48x48 realism logits.
    Receptive field ~ 94 km, so it judges storm-scale texture, not the global layout."""

    def __init__(self, cond_frames=4, out_frames=12, width=64):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        c = cond_frames + out_frames
        self.cond_frames = cond_frames
        self.net = nn.Sequential(
            sn(nn.Conv2d(c, width, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(width, 2 * width, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(2 * width, 4 * width, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(4 * width, 4 * width, 3, 1, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(4 * width, 1, 3, 1, 1)))

    def forward(self, inputs, frames):
        return self.net(torch.cat([inputs[:, -self.cond_frames:].float(), frames.float()], 1) * 2 - 1)


# ----------------------------------------------------------------------------- mixer
def parse_spec(spec):
    """'mse:1,jaccard:0.5,gan:0.3' -> {'mse': 1.0, 'jaccard': 0.5, 'gan': 0.3}. The first term is the anchor."""
    out = {}
    for part in spec.split(','):
        name, _, weight = part.strip().partition(':')
        if name not in LOSSES and name != 'gan':
            raise ValueError(f'unknown loss {name!r}; choose from {sorted(LOSSES) + ["gan"]}')
        out[name] = float(weight or 1.0)
    return out


class LossMixer:
    def __init__(self, spec, device='cuda', d_lr=2e-4, gan_ramp=500, momentum=0.98):
        self.weights = parse_spec(spec)
        self.anchor = next(iter(self.weights))
        self.norms = {}
        self.momentum, self.gan_ramp = momentum, gan_ramp
        self.disc = self.d_opt = None
        if 'gan' in self.weights:
            self.disc = PatchDiscriminator().to(device)
            self.d_opt = torch.optim.Adam(self.disc.parameters(), lr=d_lr, betas=(0.0, 0.9))

    def output_grad(self, inputs, pred, target, step):
        """Gradient of the mixed loss w.r.t. the forecast (B, T, H, W) on [0, 1], plus per-term logs."""
        p = pred.detach().float().requires_grad_(True)
        target = target.float()
        grads, logs = {}, {}
        for name in self.weights:
            if name == 'gan':
                loss = -self.disc(inputs, p).mean()
            else:
                loss = LOSSES[name](p, target)
            grads[name], = torch.autograd.grad(loss, p)
            logs[name] = float(loss)
        total = torch.zeros_like(p)
        for name, g in grads.items():
            n = float(g.norm()) + 1e-12
            self.norms[name] = n if name not in self.norms else self.momentum * self.norms[name] + (1 - self.momentum) * n
            if name == self.anchor:
                scale = self.weights[name]
            else:
                scale = self.weights[name] * self.norms[self.anchor] / self.norms[name]
                if name == 'gan':
                    scale *= min(1.0, step / max(1, self.gan_ramp))
            total += scale * g
        return total.to(pred.dtype), logs

    def disc_step(self, inputs, pred, target):
        if self.disc is None:
            return {}
        real = self.disc(inputs, target.float())
        fake = self.disc(inputs, pred.detach().float())
        loss = F.relu(1 - real).mean() + F.relu(1 + fake).mean()
        self.d_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.d_opt.step()
        return {'d_loss': float(loss), 'd_real': float(real.mean()), 'd_fake': float(fake.mean())}

    def state_dict(self):
        out = {'norms': self.norms}
        if self.disc is not None:
            out.update(disc=self.disc.state_dict(), d_opt=self.d_opt.state_dict())
        return out

    def load_state_dict(self, state):
        self.norms = state.get('norms', {})
        if self.disc is not None and 'disc' in state:
            self.disc.load_state_dict(state['disc'])
            self.d_opt.load_state_dict(state['d_opt'])
