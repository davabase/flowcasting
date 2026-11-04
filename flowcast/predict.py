"""Inference helpers: load a trained model, forecast, and roll out autoregressively.

`path` is a FlowCast checkpoint (FlowCastNet or a fine-tuned Earthformer) or the literal
'earthformer-pretrained' for the official SEVIR Earthformer weights.
"""
import numpy as np
import torch

from .model import FlowCastNet


def load_model(path, device='cuda', use_ema=True):
    if str(path) == 'earthformer-pretrained':
        from .earthformer import load_pretrained
        return load_pretrained(device)[0]
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt['config'].get('arch') == 'earthformer':
        from .earthformer import EarthformerNowcaster
        model = EarthformerNowcaster()
    else:
        model = FlowCastNet(**ckpt['config'])
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
