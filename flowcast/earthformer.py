"""Earthformer (Gao et al., NeurIPS 2022) with its official SEVIR weights, behind the FlowCast interface.

The model code is vendored unchanged from amazon-science/earth-forecasting-transformer (Apache-2.0) in
third_party/earthformer. The original PyTorch checkpoint bucket no longer allows anonymous access, so the
weights come from PaddleScience's port of the same checkpoint (`earthformer_sevir.pdparams`, published scores
identical to the paper: CSI-M 0.4419, MSE 3.6957e-3). Paddle stores Linear weights as (in, out); everything
else has the same layout as PyTorch. Earthformer was trained only on SEVIR data before 2019-06-01, so it has
never seen the June-Nov 2019 test period.
"""
import collections
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'third_party'))
from earthformer.cuboid_transformer import CuboidTransformerModel  # noqa: E402

PRETRAINED = ROOT / 'data' / 'pretrained' / 'earthformer_sevir.pdparams'
PRETRAINED_URL = 'https://paddle-org.bj.bcebos.com/paddlescience/models/earthformer/earthformer_sevir.pdparams'

# scripts/cuboid_transformer/sevir/earthformer_sevir_v1.yaml ("Configurations for pretrained checkpoint")
SEVIR_CONFIG = dict(
    input_shape=[13, 384, 384, 1], target_shape=[12, 384, 384, 1], base_units=128, block_units=None,
    scale_alpha=1.0, enc_depth=[1, 1], dec_depth=[1, 1], enc_use_inter_ffn=True, dec_use_inter_ffn=True,
    dec_hierarchical_pos_embed=False, downsample=2, downsample_type='patch_merge',
    enc_attn_patterns=['axial'] * 2, dec_self_attn_patterns=['axial'] * 2, dec_cross_attn_patterns=['cross_1x1'] * 2,
    dec_cross_last_n_frames=None, dec_use_first_self_attn=False, num_heads=4, attn_drop=0.1, proj_drop=0.1,
    ffn_drop=0.1, upsample_type='upsample', ffn_activation='gelu', gated_ffn=False, norm_layer='layer_norm',
    num_global_vectors=8, use_dec_self_global=False, dec_self_update_global=True, use_dec_cross_global=False,
    use_global_vector_ffn=False, use_global_self_attn=True, separate_global_qkv=True, global_dim_ratio=1,
    initial_downsample_type='stack_conv', initial_downsample_activation='leaky',
    initial_downsample_stack_conv_num_layers=3, initial_downsample_stack_conv_dim_list=[16, 64, 128],
    initial_downsample_stack_conv_downscale_list=[3, 2, 2], initial_downsample_stack_conv_num_conv_list=[2, 2, 2],
    padding_type='zeros', z_init_method='zeros', checkpoint_level=0, pos_embed_type='t+h+w', use_relative_pos=True,
    self_attn_use_final_proj=True, attn_linear_init_mode='0', ffn_linear_init_mode='0', conv_init_mode='0',
    down_up_linear_init_mode='0', norm_init_mode='0',
)


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """A .pdparams file is a pickle; allow only the numpy globals needed to rebuild arrays."""

    _allowed = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
                ('numpy', 'ndarray'), ('numpy', 'dtype'), ('collections', 'OrderedDict')}

    def find_class(self, module, name):
        if (module, name) not in self._allowed:
            raise pickle.UnpicklingError(f'refusing to load global {module}.{name}')
        if name == '_reconstruct':
            return np._core.multiarray._reconstruct
        return {'ndarray': np.ndarray, 'dtype': np.dtype, 'OrderedDict': collections.OrderedDict}[name]


def load_paddle_state(path=PRETRAINED):
    with open(path, 'rb') as f:
        state = _NumpyOnlyUnpickler(f).load()
    # One key in the published file carries a stray trailing newline ('...conv.bias\n'); normalise names.
    return {k.strip(): v for k, v in state.items() if isinstance(v, np.ndarray)}


def paddle_to_torch(model, paddle_state):
    """Map Paddle arrays onto the PyTorch module; transpose nn.Linear weights.

    Integer buffers (relative-position index tables) are not copied: the PyTorch module computes its own,
    and we require them to be identical to Paddle's, which checks that the port mirrors this architecture.
    """
    linear = {f'{name}.weight' for name, m in model.named_modules() if isinstance(m, nn.Linear)}
    target = model.state_dict()
    out, missing = {}, []
    for key, tensor in target.items():
        if key not in paddle_state:
            missing.append(key)
            continue
        value = paddle_state[key]
        if not np.issubdtype(value.dtype, np.floating):
            if not np.array_equal(value, tensor.cpu().numpy()):
                raise ValueError(f'{key}: index buffer differs between the Paddle port and this code')
            continue
        if key in linear:
            value = value.T
        if tuple(value.shape) != tuple(tensor.shape):
            raise ValueError(f'{key}: paddle {value.shape} vs torch {tuple(tensor.shape)}')
        out[key] = torch.from_numpy(np.ascontiguousarray(value)).to(tensor.dtype)
    unused = sorted(set(paddle_state) - set(target))
    return out, missing, unused


class EarthformerNowcaster(nn.Module):
    """(B, 13, H, W) in [0, 1] -> {'forecast': (B, 12, H, W)}, matching FlowCastNet's interface."""

    def __init__(self, checkpoint_level=0):
        super().__init__()
        self.net = CuboidTransformerModel(**{**SEVIR_CONFIG, 'checkpoint_level': checkpoint_level})
        self.config = {'arch': 'earthformer', 'checkpoint_level': checkpoint_level}

    def forward(self, x):
        y = self.net(x.unsqueeze(-1))            # NTHWC
        return {'forecast': y.squeeze(-1).float()}


def load_pretrained(device='cuda', checkpoint_level=0, path=PRETRAINED):
    model = EarthformerNowcaster(checkpoint_level)
    state, missing, unused = paddle_to_torch(model.net, load_paddle_state(path))
    # Buffers such as relative-position indices are recomputed by the module; only parameters must match.
    params = {k for k, _ in model.net.named_parameters()}
    absent = [k for k in missing if k in params]
    if absent:
        raise RuntimeError(f'pretrained checkpoint lacks parameters: {absent[:5]}')
    model.net.load_state_dict(state, strict=False)
    return model.to(device).eval(), {'missing_buffers': [k for k in missing if k not in params], 'unused': unused}
