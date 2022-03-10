import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from einops import rearrange


class LayerNorm2D(nn.Module):
    def __init__(self, dim):
        super(LayerNorm2D, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [N, C, H, W]
        # this is a naive hack
        if len(x.shape) == 4:
            n, c, h, w = x.shape
            x = rearrange(x, "n c h w -> n (h w) c")
            x = self.norm(x)
            x = rearrange(x, "n (h w) c -> n c h w", h=h)
        elif len(x.shape) == 3:
            n, c, d = x.shape
            x = self.norm(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        return x


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


def safe_cast(x, target, min=1e-15, max=1.):
    x = torch.clamp(x, min, max)
    return x.type_as(target)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def viz_single_img(x, save_path, t_stamp, cat, ind="", clamp_mean=False, raw=None):
    x = ((x - x.min()) / (x.max() - x.min())) * 255  # cast to [0, 255]
    if torch.sum(x != x).item():
        return
    if raw is not None:
        x = F.upsample_bilinear(x[None, None], size=raw.shape[-2:]).squeeze()
    x = x.unsqueeze(-1)
    if clamp_mean:
        x = torch.clamp(x, float(x.mean()) * 0.6)
        x = ((x - x.min()) / (x.max() - x.min())) * 255

    x_color = torch.cat([x, x, x], dim=-1)
    x_color = x_color.int()
    x_np = x_color.cpu().numpy()
    x_np = x_np.astype(np.uint8)
    x_np = cv2.applyColorMap(x_np, cv2.COLORMAP_JET)
    if raw is not None:
        raw = raw.permute(1, 2, 0)
        x_np = x_np * 0.44 + raw.cpu().numpy().astype(np.uint8) * 0.56
        x_np = np.clip(x_np, 0, 255)
    cv2.imwrite(f'{save_path}/{t_stamp}_{ind}_{cat}.jpg', x_np)


def visualize(batch, sum_all=-1, per_ch=True, save_path="viz/", cat="", t_stamp=0, raw=None):
    for i in range(len(batch)):
        if sum_all >= 0:
            sum_batch = torch.sum(batch[i], dim=sum_all)
            for c in range(sum_batch.shape[0]):
                viz_single_img(sum_batch[c], save_path, t_stamp, cat,
                               ind=f"b{i}g{c}", raw=raw[i])
        elif per_ch:
            if len(batch.shape) == 4:
                B, C, H, W = batch.shape
                for c in range(C):
                    viz_single_img(batch[i, c, :, :], save_path, t_stamp, cat,
                                   ind=f"b{i}c{c}", raw=raw[i])
            elif len(batch.shape) == 5:
                B, G, C, H, W = batch.shape
                for g in range(G):
                    for c in range(C):
                        viz_single_img(batch[i, g, c, :, :], save_path, t_stamp, cat,
                                       ind=f"b{i}g{g}c{c}", raw=raw[i])
            else:
                raise NotImplementedError
