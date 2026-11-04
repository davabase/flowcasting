"""Nowcasting verification metrics, accumulated over a whole test set (SEVIR / Earthformer convention).

CSI, POD, SUCC (1 - FAR) and BIAS are computed from hits / misses / false alarms *summed* over
every pixel and sample before dividing, per threshold and lead time. CSI-M is the mean CSI over the
six standard SEVIR VIL thresholds. Pooled variants max-pool both fields first, which rewards
forecasts that put storms in roughly the right place (scale-aware verification).
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

THRESHOLDS = (16, 74, 133, 160, 181, 219)
POOLS = (1, 4, 16)


class Scores:
    def __init__(self, lead_times, thresholds=THRESHOLDS, pools=POOLS, device='cuda'):
        self.thresholds, self.pools, self.leads = thresholds, pools, lead_times
        shape = (len(pools), len(thresholds), lead_times)
        self.hits = torch.zeros(shape, dtype=torch.float64, device=device)
        self.misses = torch.zeros_like(self.hits)
        self.false_alarms = torch.zeros_like(self.hits)
        self.sq_err = torch.zeros(lead_times, dtype=torch.float64, device=device)
        self.abs_err = torch.zeros_like(self.sq_err)
        self.pixels = 0

    @torch.no_grad()
    def update(self, pred, target):
        """pred, target: (B, T, H, W) tensors on the raw SEVIR 0-255 scale."""
        pred, target = pred.float().clamp(0, 255), target.float()
        diff = (pred - target) / 255.0
        self.sq_err += (diff ** 2).sum(dim=(0, 2, 3)).double()
        self.abs_err += diff.abs().sum(dim=(0, 2, 3)).double()
        self.pixels += pred.shape[0] * pred.shape[2] * pred.shape[3]
        for p, pool in enumerate(self.pools):
            pp = F.max_pool2d(pred, pool) if pool > 1 else pred
            tt = F.max_pool2d(target, pool) if pool > 1 else target
            for k, threshold in enumerate(self.thresholds):
                a, b = pp >= threshold, tt >= threshold
                self.hits[p, k] += (a & b).sum(dim=(0, 2, 3)).double()
                self.misses[p, k] += (~a & b).sum(dim=(0, 2, 3)).double()
                self.false_alarms[p, k] += (a & ~b).sum(dim=(0, 2, 3)).double()

    def result(self):
        h, m, f = self.hits, self.misses, self.false_alarms
        out = {'thresholds': list(self.thresholds), 'pools': list(self.pools)}

        def ratio(a, b):
            return (a / b.clamp(min=1)).cpu().numpy()

        for name, (hh, mm, ff) in {'per_lead': (h, m, f),
                                   'overall': (h.sum(-1), m.sum(-1), f.sum(-1))}.items():
            block = {'csi': ratio(hh, hh + mm + ff), 'pod': ratio(hh, hh + mm),
                     'succ': ratio(hh, hh + ff), 'bias': ratio(hh + ff, hh + mm)}
            out[name] = {k: v.tolist() for k, v in block.items()}
            out[name]['csi_m'] = block['csi'].mean(axis=1).tolist()   # (pools,) or (pools, leads)
        out['per_lead']['mse'] = (self.sq_err / self.pixels).cpu().numpy().tolist()
        out['per_lead']['mae'] = (self.abs_err / self.pixels).cpu().numpy().tolist()
        out['overall']['mse'] = float(self.sq_err.sum() / (self.pixels * self.leads))
        out['overall']['mae'] = float(self.abs_err.sum() / (self.pixels * self.leads))
        return out


def summary_row(result):
    """Flat dict of the headline numbers for one method."""
    o = result['overall']
    row = {'CSI-M': o['csi_m'][0]}
    for k, t in enumerate(result['thresholds']):
        row[f'CSI-{t}'] = o['csi'][0][k]
    for p, pool in enumerate(result['pools'][1:], 1):
        row[f'CSI-M-pool{pool}'] = o['csi_m'][p]
    row['BIAS-133'] = o['bias'][0][result['thresholds'].index(133)]
    row['MSE(1e-3)'] = o['mse'] * 1e3
    row['MAE(1e-3)'] = o['mae'] * 1e3
    return row


def save(path, results):
    with open(path, 'w') as f:
        json.dump(results, f, indent=1)
