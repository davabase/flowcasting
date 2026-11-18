"""Realism / sharpness verification that pixel scores cannot see.

* Power spectrum: radially averaged power of forecasts vs observations, per lead time. The ratio at small
  wavelengths says how much fine-scale detail survives (1.0 = as detailed as real radar; a blurry forecast
  falls towards 0). "Effective resolution" is the shortest wavelength down to which the forecast keeps at
  least half of the observed power, scanning from large to small scales.
* FSS (fractions skill score, Roberts & Lean 2008): compares the *fraction* of pixels above a threshold in
  neighbourhoods of n x n km, which removes the double penalty: 1 = perfect, 0 = no skill.
* Intensity histogram: how often each VIL value occurs, forecast vs observed (blur removes the tail).
"""
import numpy as np
import torch
import torch.nn.functional as F

from .losses import radial_power

FSS_THRESHOLDS = (74, 133, 181)
FSS_WINDOWS = (1, 5, 17, 33)       # km (odd, so neighbourhoods are centred)
SMALL_SCALES = (4, 16)             # km: wavelength band summarised as "small-scale power"


def box_mean(x, n):
    """Mean over the n x n neighbourhood clipped to the image (== avg_pool2d(x, n, 1, n // 2,
    count_include_pad=False) for odd n), via a summed-area table: O(1) per pixel instead of O(n^2)."""
    h, w = x.shape[-2:]
    r = n // 2
    table = F.pad(x.float(), (1, 0, 1, 0)).cumsum(-1).cumsum(-2)
    rows, cols = torch.arange(h, device=x.device), torch.arange(w, device=x.device)
    r0, r1 = (rows - r).clamp(min=0), (rows + r + 1).clamp(max=h)
    c0, c1 = (cols - r).clamp(min=0), (cols + r + 1).clamp(max=w)

    def at(ri, ci):
        return table.index_select(-2, ri).index_select(-1, ci)
    total = at(r1, c1) - at(r0, c1) - at(r1, c0) + at(r0, c0)
    return total / ((r1 - r0)[:, None] * (c1 - c0)[None, :]).float()


class Sharpness:
    def __init__(self, lead_times, device='cuda', size=384):
        self.leads, self.size = lead_times, size
        nb = size // 2 + 1
        self.psd_pred = torch.zeros(lead_times, nb, dtype=torch.float64, device=device)
        self.psd_obs = torch.zeros_like(self.psd_pred)
        self.hist_pred = torch.zeros(256, dtype=torch.float64, device=device)
        self.hist_obs = torch.zeros_like(self.hist_pred)
        shape = (len(FSS_THRESHOLDS), len(FSS_WINDOWS), lead_times)
        self.fss_num = torch.zeros(shape, dtype=torch.float64, device=device)
        self.fss_den = torch.zeros_like(self.fss_num)
        self.samples = 0

    @torch.no_grad()
    def update(self, pred, target):
        """pred, target: (B, T, H, W) on the 0-255 scale."""
        pred, target = pred.float().clamp(0, 255), target.float()
        b, t = pred.shape[:2]
        self.psd_pred += radial_power(pred.flatten(0, 1) / 255).view(b, t, -1).sum(0).double()
        self.psd_obs += radial_power(target.flatten(0, 1) / 255).view(b, t, -1).sum(0).double()
        self.hist_pred += torch.bincount(pred.round().long().flatten(), minlength=256).double()
        self.hist_obs += torch.bincount(target.round().long().clamp(0, 255).flatten(), minlength=256).double()
        for i, thr in enumerate(FSS_THRESHOLDS):
            a, o = (pred >= thr).float(), (target >= thr).float()
            for j, n in enumerate(FSS_WINDOWS):
                if n > 1:
                    fa, fo = box_mean(a, n), box_mean(o, n)
                else:
                    fa, fo = a, o
                self.fss_num[i, j] += ((fa - fo) ** 2).sum(dim=(0, 2, 3)).double()
                self.fss_den[i, j] += (fa ** 2 + fo ** 2).sum(dim=(0, 2, 3)).double()
        self.samples += b

    def result(self):
        ratio = (self.psd_pred / self.psd_obs.clamp(min=1e-30)).cpu().numpy()       # (leads, nb)
        k = np.arange(ratio.shape[1])
        wavelength = np.where(k > 0, self.size / np.maximum(k, 1), np.inf)
        eff = []
        for lead in range(self.leads):
            below = np.flatnonzero((ratio[lead] < 0.5) & (k >= 1))
            eff.append(float(wavelength[below[0]]) if len(below) else 2.0)
        band = (wavelength >= SMALL_SCALES[0]) & (wavelength <= SMALL_SCALES[1])
        small = (self.psd_pred[:, band].sum(1) / self.psd_obs[:, band].sum(1)).cpu().numpy()
        fss_lead = (1 - self.fss_num / self.fss_den.clamp(min=1e-30)).cpu().numpy()
        fss_all = (1 - self.fss_num.sum(-1) / self.fss_den.sum(-1).clamp(min=1e-30)).cpu().numpy()
        return {
            'wavelength_km': [float(w) if np.isfinite(w) else None for w in wavelength],
            'psd_ratio': ratio.tolist(),                      # (leads, bins)
            'psd_obs': (self.psd_obs / max(1, self.samples)).cpu().numpy().tolist(),
            'psd_pred': (self.psd_pred / max(1, self.samples)).cpu().numpy().tolist(),
            'effective_resolution_km': eff,                   # per lead
            'small_scale_power': small.tolist(),              # per lead, 4-16 km band, pred / obs
            'fss_thresholds': list(FSS_THRESHOLDS), 'fss_windows_km': list(FSS_WINDOWS),
            'fss': fss_all.tolist(), 'fss_per_lead': fss_lead.tolist(),
            'hist_pred': (self.hist_pred / self.hist_pred.sum()).cpu().numpy().tolist(),
            'hist_obs': (self.hist_obs / self.hist_obs.sum()).cpu().numpy().tolist(),
        }


def sharp_row(result):
    """Headline sharpness numbers: 60-min effective resolution and small-scale power, FSS-133 at 17 km."""
    s = result['sharp']
    i, j = s['fss_thresholds'].index(133), s['fss_windows_km'].index(17)
    return {'EffRes60(km)': s['effective_resolution_km'][-1], 'SmallPow30': s['small_scale_power'][5],
            'SmallPow60': s['small_scale_power'][-1], 'FSS133@17km': s['fss'][i][j]}
