import math
import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
import logging
import typing as tp

logger = logging.getLogger(__name__)

class SubjectLayers(nn.Module):
    """Per subject linear layer."""
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5

    def forward(self, x, subjects):
        _, C, D = self.weights.shape
        w = self.weights.clone()
        weights = w.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D)) 
        x = torch.einsum("bct,bcd->bdt", x, weights)
        return x

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"

class ConvSequence(nn.Module):

    def __init__(self, channels: tp.Sequence[int], kernel: int = 4, dilation_growth: int = 1,
                 dilation_period: tp.Optional[int] = None, stride: int = 2,
                 dropout: float = 0.0, leakiness: float = 0.0, groups: int = 1,
                 decode: bool = False, batch_norm: bool = False, dropout_input: float = 0,
                 skip: bool = False, scale: tp.Optional[float] = None, rewrite: bool = False,
                 activation_on_last: bool = True, post_skip: bool = False, glu: int = 0,
                 glu_context: int = 0, glu_glu: bool = True, activation: tp.Any = None) -> None:
        super().__init__()
        dilation = 1
        self.leakiness = leakiness
        channels = tuple(channels)
        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        if activation is None:
            activation = partial(nn.LeakyReLU, leakiness)
        Conv = nn.Conv1d if not decode else nn.ConvTranspose1d
        # build layers
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            # Set dropout for the input of the conv sequence if defined
            if k == 0 and dropout_input:
                assert 0 < dropout_input < 1
                layers.append(nn.Dropout(dropout_input))

            # conv layer
            if dilation_growth > 1:
                assert kernel % 2 != 0, "Supports only odd kernel with dilation for now"
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1
            pad = kernel // 2 * dilation
            layers.append(Conv(chin, chout, kernel, stride, pad,
                               dilation=dilation, groups=groups if k > 0 else 1))
            dilation = dilation * dilation_growth
            # non-linearity
            if activation_on_last or not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())
                if dropout:
                    layers.append(nn.Dropout(dropout))
                if rewrite:
                    layers += [nn.Conv1d(chout, chout, 1), nn.LeakyReLU(leakiness)]
                    # layers += [nn.Conv1d(chout, 2 * chout, 1), nn.GLU(dim=1)]
            if chin == chout and skip:
                if scale is not None:
                    layers.append(LayerScale(chout, scale))
                if post_skip:
                    layers.append(Conv(chout, chout, 1, groups=chout, bias=False))

            self.sequence.append(nn.Sequential(*layers))
            if glu and (k + 1) % glu == 0:
                ch = 2 * chout if glu_glu else chout
                act = nn.GLU(dim=1) if glu_glu else activation()
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, ch, 1 + 2 * glu_context, padding=glu_context), act))
            else:
                self.glus.append(None)

    def forward(self, x: tp.Any, functional=False) -> tp.Any:
        for module_idx, module in enumerate(self.sequence):
            old_x = x
            if functional:
                for layer in module:
                    if isinstance(layer, nn.Dropout):
                        x = F.dropout(x, p=layer.dropout_p)
                    elif isinstance(layer, nn.Conv1d):
                        # Clone the weights and biases
                        weight = layer.weight.clone()
                        bias = layer.bias.clone() if layer.bias is not None else None
                        # Perform the convolution using the functional API
                        x = F.conv1d(x, weight, bias=bias, stride=layer.stride, padding=layer.padding, 
                            dilation=layer.dilation, groups=layer.groups)
                    elif isinstance(layer, nn.ConvTranspose1d):
                        # Clone weights and biases
                        weight = layer.weight.clone()
                        bias = layer.bias.clone() if layer.bias is not None else None
                        # Perform the transposed convolution using the functional API
                        x = F.conv_transpose1d(x, weight, bias=bias, stride=layer.stride,
                                    padding=layer.padding, output_padding=layer.output_padding,
                                    groups=layer.groups, dilation=layer.dilation)
                    elif isinstance(layer, nn.BatchNorm1d):
                        # Clone running mean and variance
                        running_mean = layer.running_mean.clone() if layer.track_running_stats else None
                        running_var = layer.running_var.clone() if layer.track_running_stats else None
                        # Clone weight and bias
                        weight = layer.weight.clone() if layer.affine else None
                        bias = layer.bias.clone() if layer.affine else None
                        # Perform batch normalization using the functional API
                        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, 
                                training=layer.training or not layer.track_running_stats, 
                                momentum=layer.momentum, eps=layer.eps)
                    elif isinstance(layer, nn.GELU):
                        x = F.gelu(x)
                    elif isinstance(layer, nn.LeakyReLU):
                        x = F.leaky_relu(x, self.leakiness)
                    else:
                        x = layer(x)
            else:
                x = module(x)

            if self.skip and x.shape == old_x.shape:
                x = x + old_x
            glu = self.glus[module_idx]
            if glu is not None:
                if functional:
                    for layer in glu:
                        if isinstance(layer, nn.GLU):
                            x = F.glu(x, dim=1)
                        elif isinstance(layer, nn.Conv1d):
                            # Clone the weights and biases
                            weight = layer.weight.clone()
                            bias = layer.bias.clone() if layer.bias is not None else None
                            # Perform the convolution using the functional API
                            x = F.conv1d(x, weight, bias=bias, stride=layer.stride, padding=layer.padding, 
                                dilation=layer.dilation, groups=layer.groups)
                        elif isinstance(layer, nn.GELU):
                            x = F.gelu(x)
                        elif isinstance(layer, nn.LeakyReLU):
                            x = F.leaky_relu(x, self.leakiness)
                else:
                    x = glu(x)
        return x
    
class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """
    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init / boost
        self.boost = boost

    def forward(self, x):
        return (self.boost * self.scale[:, None]) * x

class ChannelMerger(nn.Module):
    def __init__(self, chout: int, pos_dim: int = 256,
                 dropout: float = 0, usage_penalty: float = 0.):
        super().__init__()
        assert pos_dim % 4 == 0
        self.heads = nn.Parameter(torch.randn(chout, pos_dim, requires_grad=True))
        self.heads.data /= pos_dim ** 0.5
        self.dropout = dropout
        self.embedding = FourierEmb(pos_dim)
        self.usage_penalty = usage_penalty
        self._penalty = torch.tensor(0.)
        self.warn_no_positions = False

    @property
    def training_penalty(self):
        return self._penalty.to(next(self.parameters()).device)

    def forward(self, meg, sensor_xyz):
        B, C, T = meg.shape
        meg = meg.clone()

        # Extract and normalize positions
        positions = sensor_xyz.squeeze(0)[:, :2].float() # (Channel, x/y)

        # Check if positions contains NaNs and ignore if not present
        if torch.isnan(positions).any():
            if not self.warn_no_positions:
                print("Warning: NaNs in positions, setting to -0.1")
                self.warn_no_positions = True
            positions = torch.zeros_like(positions, device=positions.device) - 0.1
        else:
            x = positions[:, 0]
            y = positions[:, 1]
            x = (x - x.min()) / (x.max() - x.min())
            y = (y - y.min()) / (y.max() - y.min())
            positions[:, 0] = x
            positions[:, 1] = y

        positions = positions.unsqueeze(0).expand(B, -1, -1)

        embedding = self.embedding(positions)
        score_offset = torch.zeros(B, C, device=meg.device)
        # score_offset[self.position_getter.is_invalid(positions)] = float('-inf')

        if self.training and self.dropout:
            center_to_ban = torch.rand(2, device=meg.device)
            radius_to_ban = self.dropout
            banned = (positions - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset[banned] = float('-inf')

        heads = self.heads[None].expand(B, -1, -1)

        scores = torch.einsum("bcd,bod->boc", embedding, heads)
        scores = scores + score_offset[:, None]
        weights = torch.softmax(scores, dim=2)
        out = torch.einsum("bct,boc->bot", meg, weights)
        if self.training and self.usage_penalty > 0.:
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = self.usage_penalty * usage
        return out


class FourierEmb(nn.Module):
    """
    Fourier positional embedding.
    Unlike trad. embedding this is not using exponential periods
    for cosines and sines, but typical `2 pi k` which can represent
    any function over [0, 1]. As this function would be necessarily periodic,
    we take a bit of margin and do over [-0.2, 1.2].
    """
    def __init__(self, dimension: int = 256, margin: float = 0.2):
        super().__init__()
        n_freqs = (dimension // 2)**0.5
        assert int(n_freqs ** 2 * 2) == dimension
        self.dimension = dimension
        self.margin = margin

    def forward(self, positions):
        *O, D = positions.shape
        assert D == 2
        *O, D = positions.shape
        n_freqs = (self.dimension // 2)**0.5
        freqs_y = torch.arange(n_freqs).to(positions)
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        positions = positions[..., None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y).view(*O, -1)
        emb = torch.cat([
            torch.cos(loc),
            torch.sin(loc),
        ], dim=-1)
        return emb