"""Generative texture refiner: a small noise-conditioned GAN on top of a frozen deterministic forecast.

Cascade in the spirit of NowcastNet (Zhang et al., Nature 2023: deterministic evolution network + noise-driven
generative network trained adversarially with pool regularisation), DGMR (Ravuri et al., Nature 2021) and
CasCast (Gong et al., ICML 2024: deterministic model + generative refiner). Kept small enough for a laptop GPU.

    base forecast (12 frames) + last 4 observed frames + noise at 1/8, 1/4, 1/2 resolution
      -> U-Net (bilinear upsampling, no transposed conv / pixel shuffle, so no checkerboards)
      -> residual added to the base forecast

Because each call draws new noise, the refiner produces *samples*: different plausible small-scale textures
for the same large-scale forecast, instead of their blurry average.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
                                  nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class TextureGenerator(nn.Module):
    def __init__(self, frames=12, cond=4, widths=(32, 64, 128, 192), noise=(8, 8, 16)):
        super().__init__()
        self.frames, self.cond, self.noise = frames, cond, noise
        w0, w1, w2, w3 = widths
        self.inp = nn.Conv2d(frames + cond, w0, 3, padding=1)
        self.e0, self.e1, self.e2 = ResBlock(w0), ResBlock(w1), ResBlock(w2)
        self.down0 = nn.Conv2d(w0, w1, 3, stride=2, padding=1)
        self.down1 = nn.Conv2d(w1, w2, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(w2, w3, 3, stride=2, padding=1)
        # noise enters at 1/8 (48 px), 1/4 and 1/2 resolution
        self.mid_in = nn.Conv2d(w3 + noise[2], w3, 3, padding=1)
        self.mid = nn.Sequential(ResBlock(w3), ResBlock(w3))
        self.up2 = nn.Conv2d(w3 + w2 + noise[1], w2, 3, padding=1)
        self.d2 = ResBlock(w2)
        self.up1 = nn.Conv2d(w2 + w1 + noise[0], w1, 3, padding=1)
        self.d1 = ResBlock(w1)
        self.up0 = nn.Conv2d(w1 + w0, w0, 3, padding=1)
        self.d0 = ResBlock(w0)
        self.out = nn.Conv2d(w0, frames, 3, padding=1)
        nn.init.zeros_(self.out.weight)       # starts as the identity on the base forecast
        nn.init.zeros_(self.out.bias)

    def forward(self, base, context, generator=None):
        """base (B, 12, H, W), context (B, 4, H, W) on [0, 1] -> refined (B, 12, H, W)."""
        b, _, h, w = base.shape

        def z(ch, scale):
            return torch.randn(b, ch, h // scale, w // scale, device=base.device, dtype=base.dtype, generator=generator)
        up = lambda t: F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)   # noqa: E731
        x0 = self.e0(self.inp(torch.cat([base, context], 1)))
        x1 = self.e1(self.down0(x0))
        x2 = self.e2(self.down1(x1))
        x3 = self.mid(self.mid_in(torch.cat([self.down2(x2), z(self.noise[2], 8)], 1)))
        y2 = self.d2(self.up2(torch.cat([up(x3), x2, z(self.noise[1], 4)], 1)))
        y1 = self.d1(self.up1(torch.cat([up(y2), x1, z(self.noise[0], 2)], 1)))
        y0 = self.d0(self.up0(torch.cat([up(y1), x0], 1)))
        return base + self.out(y0)


class TextureDiscriminator(nn.Module):
    """Conditional PatchGAN that sees each forecast frame twice: as is, and high-passed (frame minus its ~16 km
    blur, amplified). Texture is a small signal next to the large-scale field; the high-pass view makes a missing
    or fake texture obvious, so the discriminator learns quickly what real radar detail looks like."""

    def __init__(self, cond_frames=4, out_frames=12, width=64, gain=4.0):
        super().__init__()
        from .losses import PatchDiscriminator
        self.net = PatchDiscriminator(cond_frames, 2 * out_frames, width)
        self.gain = gain

    def forward(self, inputs, frames):
        from .losses import _lowpass
        frames = frames.float()
        return self.net(inputs, torch.cat([frames, 0.5 + self.gain * (frames - _lowpass(frames, 2.0))], 1))


class RefinedModel(nn.Module):
    """Frozen base forecaster + texture refiner behind the FlowCast model interface (one sample per call).
    A fixed seed makes evaluation reproducible: the n-th call always uses the same noise."""

    def __init__(self, base, refiner, seed=0):
        super().__init__()
        self.base, self.refiner = base, refiner
        self.config = {'arch': getattr(base, 'config', {}).get('arch', 'refined')}
        self.seed = seed
        self._gen = None

    def forward(self, x):
        if self._gen is None:
            self._gen = torch.Generator(device=x.device).manual_seed(self.seed)
        base = self.base(x)['forecast'].float()
        return {'forecast': self.refiner(base, x[:, -self.refiner.cond:].float(), generator=self._gen)}
