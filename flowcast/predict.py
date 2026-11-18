"""Inference helpers: load a trained model, forecast, and roll out autoregressively.

`path` is a FlowCast checkpoint (FlowCastNet or a fine-tuned Earthformer) or the literal
'earthformer-pretrained' for the official SEVIR Earthformer weights.
"""
import inspect

import numpy as np
import torch

from .model import FlowCastNet


def model_kwargs(config):
    """Constructor arguments from a checkpoint config (which may also record e.g. the training loss)."""
    params = inspect.signature(FlowCastNet.__init__).parameters
    return {k: v for k, v in config.items() if k in params}


class ScaleBlend(torch.nn.Module):
    """Large scales (> ~16 km) from one model, small-scale detail from another: lowpass(a) + (b - lowpass(b)).

    Classic nowcasting scale decomposition, used here to pair Earthformer's large-scale skill with the
    fine-scale texture of a sharpened FlowCastNet. No training involved."""

    def __init__(self, large, small):
        super().__init__()
        self.large, self.small = large, small
        self.config = {'arch': 'earthformer' if getattr(large, 'config', {}).get('arch') == 'earthformer' else 'blend'}

    def forward(self, x):
        from .losses import _lowpass
        a = self.large(x)['forecast'].float()
        out = self.small(x)
        b = out['forecast'].float()
        if 'valid' in out:      # no data flows in across the domain edge: fill from the large-scale model there
            b = out['valid'] * b + (1 - out['valid']) * a
        return {'forecast': _lowpass(a) + b - _lowpass(b)}


class OpticalFlowModel(torch.nn.Module):
    """Optical-flow extrapolation (flowcast.optical_flow, CPU) behind the model interface, for use in a ScaleBlend."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)   # gives the module a device
        self.config = {'arch': 'optical_flow'}

    def forward(self, x):
        from .optical_flow import optical_flow_valid
        frames = (x.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        fc, valid = optical_flow_valid(frames)
        return {'forecast': torch.as_tensor(fc, device=x.device).float() / 255,
                'valid': torch.as_tensor(valid, device=x.device)}


def load_model(path, device='cuda', use_ema=True):
    if str(path) == 'optical-flow':
        return OpticalFlowModel().to(device)
    if str(path).startswith('blend:'):          # 'blend:<large-scale source>|<small-scale source>'
        large, small = str(path)[6:].split('|')
        return ScaleBlend(load_model(large, device, use_ema), load_model(small, device, use_ema)).eval()
    if str(path) == 'earthformer-pretrained':
        from .earthformer import load_pretrained
        return load_pretrained(device)[0]
    seed = 0
    if str(path).count('@') == 1 and str(path).rsplit('@', 1)[1].isdigit():   # 'refiner.pt@3': noise seed 3
        path, seed = str(path).rsplit('@', 1)
        seed = int(seed)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt['config'].get('arch') == 'refiner':      # generative texture refiner on a frozen base forecaster
        from .refiner import RefinedModel, TextureGenerator
        gen = TextureGenerator().to(device)
        gen.load_state_dict(ckpt['ema' if use_ema else 'gen'])
        return RefinedModel(load_model(ckpt['config']['base'], device, use_ema), gen.eval(), seed=seed).eval()
    if ckpt['config'].get('arch') == 'earthformer':
        from .earthformer import EarthformerNowcaster
        model = EarthformerNowcaster()
    else:
        model = FlowCastNet(**model_kwargs(ckpt['config']))
    model.load_state_dict(ckpt['ema' if use_ema else 'model'])
    return model.to(device).to(memory_format=torch.channels_last).eval()


@torch.no_grad()
def forecast(model, inputs, batch=4, return_parts=False):
    """inputs: uint8 array/tensor (B, 13, H, W). Yields float tensors on the 0-255 scale per mini-batch.

    Returns a list of dicts with 'forecast' (b, 12, H, W) and, if requested, 'flow' and 'residual'.
    """
    device = next(model.parameters()).device
    outputs = []
    for i in range(0, len(inputs), batch):
        x = torch.as_tensor(np.asarray(inputs[i:i + batch])).to(device).float().div_(255)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x.contiguous(memory_format=torch.channels_last))
        item = {'forecast': (out['forecast'] * 255).clamp_(0, 255)}
        if return_parts:
            item['flow'] = out['flow']
            item['residual'] = out['residual'] * 255
            item['advected'] = out['advected'] * 255
        outputs.append(item)
    return outputs


@torch.no_grad()
def rollout(model, inputs, steps, batch=4):
    """Autoregressive simulation: feed the model's own forecasts back as context until `steps` frames.

    inputs uint8 (B, 13, H, W) -> float32 numpy (B, steps, H, W) on the 0-255 scale.
    """
    context = np.asarray(inputs).astype(np.float32)
    produced = []
    total = 0
    while total < steps:
        pred = torch.cat([o['forecast'] for o in forecast(model, np.clip(np.rint(context), 0, 255).astype(np.uint8),
                                                          batch)]).float().cpu().numpy()
        produced.append(pred)
        total += pred.shape[1]
        context = np.concatenate([context, pred], axis=1)[:, -inputs.shape[1]:]
    return np.concatenate(produced, axis=1)[:, :steps]
