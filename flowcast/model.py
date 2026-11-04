"""FlowCastNet: a neural advection-plus-growth model for radar nowcasting.

The past `in_frames` VIL frames go through a ConvNeXt U-Net that runs on a 4x space-to-depth
latent grid (384 -> 96). For every lead time the network emits

* a smooth displacement field d_t (latent resolution, bilinearly upsampled) that warps the most
  recent observation:  advected_t(p) = x_last(p + d_t(p))   -- a learned semi-Lagrangian step
* an intensity residual r_t (full resolution via pixel shuffle, then a light full-resolution
  refinement conv) for growth, decay and initiation

    forecast_t = advected_t + r_t

Both heads are zero-initialised, so an untrained model is exactly the persistence forecast and
training only has to learn the corrections. The fields are returned for visualisation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, expansion=4, layer_scale=1e-2):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels * expansion, 1)
        self.pw2 = nn.Conv2d(channels * expansion, channels, 1)
        self.gamma = nn.Parameter(torch.full((channels, 1, 1), layer_scale))

    def forward(self, x):
        y = self.pw2(F.gelu(self.pw1(self.norm(self.dwconv(x)))))
        return x + self.gamma * y


def stage(channels, depth):
    return nn.Sequential(*[ConvNeXtBlock(channels) for _ in range(depth)])


def warp(image, displacement):
    """image (B, 1, H, W); displacement (B, T, 2, H, W) in pixels -> (B, T, H, W) sampled at p + d(p)."""
    b, t, _, h, w = displacement.shape
    ys = torch.linspace(-1, 1, h, device=image.device)
    xs = torch.linspace(-1, 1, w, device=image.device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([gx + displacement[:, :, 0] * (2 / (w - 1)),
                        gy + displacement[:, :, 1] * (2 / (h - 1))], dim=-1)
    source = image.expand(b, t, h, w).reshape(b * t, 1, h, w)
    out = F.grid_sample(source.float(), grid.reshape(b * t, h, w, 2).float(),
                        mode='bilinear', padding_mode='zeros', align_corners=True)
    return out.view(b, t, h, w)


class FlowCastNet(nn.Module):
    def __init__(self, in_frames=13, out_frames=12, widths=(96, 192, 384, 384), depths=(2, 2, 4, 2),
                 patch=4, flow_scale=8.0):
        super().__init__()
        self.in_frames, self.out_frames, self.patch, self.flow_scale = in_frames, out_frames, patch, flow_scale
        self.config = dict(in_frames=in_frames, out_frames=out_frames, widths=list(widths),
                           depths=list(depths), patch=patch, flow_scale=flow_scale)
        self.stem = nn.Sequential(nn.PixelUnshuffle(patch),
                                  nn.Conv2d(in_frames * patch * patch, widths[0], 3, padding=1),
                                  LayerNorm2d(widths[0]))
        self.encoders = nn.ModuleList(stage(w, d) for w, d in zip(widths[:-1], depths[:-1]))
        self.downs = nn.ModuleList(nn.Sequential(LayerNorm2d(a), nn.Conv2d(a, b, 2, stride=2))
                                   for a, b in zip(widths[:-1], widths[1:]))
        self.bottleneck = stage(widths[-1], depths[-1])
        self.ups = nn.ModuleList(nn.ConvTranspose2d(b, a, 2, stride=2) for a, b in zip(widths[:-1], widths[1:]))
        self.fuses = nn.ModuleList(nn.Conv2d(2 * w, w, 1) for w in widths[:-1])
        self.decoders = nn.ModuleList(stage(w, max(1, d // 2)) for w, d in zip(widths[:-1], depths[:-1]))
        self.head_norm = LayerNorm2d(widths[0])
        self.flow_head = nn.Conv2d(widths[0], out_frames * 2, 3, padding=1)
        self.residual_head = nn.Conv2d(widths[0], out_frames * patch * patch, 3, padding=1)
        # Light full-resolution refinement over [advected, residual] removes pixel-shuffle block artifacts.
        self.refine = nn.Sequential(nn.Conv2d(2 * out_frames, 32, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(32, out_frames, 3, padding=1))
        for head in (self.flow_head, self.residual_head, self.refine[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x):
        """x: (B, in_frames, H, W) scaled to [0, 1]. Returns a dict with the forecast and its parts."""
        b, _, h, w = x.shape
        z = self.stem(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            z = encoder(z)
            skips.append(z)
            z = down(z)
        z = self.bottleneck(z)
        for up, fuse, decoder, skip in reversed(list(zip(self.ups, self.fuses, self.decoders, skips))):
            z = decoder(fuse(torch.cat([up(z), skip], dim=1)))
        z = self.head_norm(z)
        flow_latent = self.flow_head(z).float() * self.flow_scale          # pixels at full resolution
        flow = F.interpolate(flow_latent, scale_factor=self.patch, mode='bilinear', align_corners=False)
        flow = flow.view(b, self.out_frames, 2, h, w)
        residual = F.pixel_shuffle(self.residual_head(z), self.patch).float()
        advected = warp(x[:, -1:], flow)
        residual = residual + self.refine(torch.cat([advected, residual], dim=1).to(z.dtype)).float()
        return {'forecast': advected + residual, 'advected': advected, 'residual': residual,
                'flow': flow, 'flow_latent': flow_latent.view(b, self.out_frames, 2, h // self.patch, w // self.patch)}


def build(config=None):
    return FlowCastNet(**(config or {}))
