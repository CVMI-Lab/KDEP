import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .misc import trunc_normal_


class FullRelPos(nn.Module):
    def __init__(self, h, w, dim, drop_ratio=0., norm_roi=True, has_map=False, learn_map=True, winh=0, winw=0):
        super(FullRelPos, self).__init__()
        self.h, self.w = h, w
        self.norm_roi = norm_roi
        self.rel_emb_h = nn.Parameter(torch.Tensor(2 * h - 1, dim // 2))  # [-(q-1), q-1]
        self.rel_emb_w = nn.Parameter(torch.Tensor(2 * w - 1, dim // 2))  # [-(q-1), q-1]
        self.has_map = has_map

        if has_map:
            coor_map_h = torch.zeros(h, h, 2 * h - 1)  # [qh, kh, rh]
            coor_map_w = torch.zeros(w, w, 2 * w - 1)  # [qw, kw, rw]
            rel_idx_h = torch.arange(2 * h - 1)
            rel_idx_w = torch.arange(2 * w - 1)
            winh = winh if winh else 2 * h - 1
            winw = winw if winw else 2 * w - 1
            coor_map_h = self._init_coor_map(h, rel_idx_h, h-1, winh, coor_map_h)
            coor_map_w = self._init_coor_map(w, rel_idx_w, w-1, winw, coor_map_w)
            self.coor_map_h = nn.Parameter(coor_map_h, requires_grad=True) if learn_map else coor_map_h
            self.coor_map_w = nn.Parameter(coor_map_w, requires_grad=True) if learn_map else coor_map_w
        else:
            # get relative coordinates of the q-k index table
            coords_h = torch.arange(h)
            coords_w = torch.arange(w)
            self.rel_idx_h = coords_h[None, :] - coords_h[:, None]
            self.rel_idx_w = coords_w[None, :] - coords_w[:, None]
            self.rel_idx_h += h - 1
            self.rel_idx_w += w - 1

        nn.init.normal_(self.rel_emb_w, std=dim ** -0.5)
        nn.init.normal_(self.rel_emb_h, std=dim ** -0.5)
        trunc_normal_(self.rel_emb_w, std=.02)
        trunc_normal_(self.rel_emb_h, std=.02)

        self.drop_ratio = drop_ratio

    def _init_coor_map(self, l, rel_idx, abs_len, win, coor_map):
        for q in range(l):
            for k in range(l):
                for r in rel_idx:
                    offset = q - k + abs_len
                    if offset == r and offset <= win:
                        coor_map[q, k, r] = 1
        return coor_map

    def forward(self, q, attn, rois=None):
        if self.has_map:
            coor_map_h = self.coor_map_h.to(q.device)
            coor_map_w = self.coor_map_w.to(q.device)
            coor_map_h = F.sigmoid(coor_map_h)
            coor_map_w = F.sigmoid(coor_map_w)
            abs_pos_h = torch.einsum("q k r, r c -> q k c", coor_map_h, self.rel_emb_h)
            abs_pos_w = torch.einsum("q k r, r c -> q k c", coor_map_w, self.rel_emb_w)
        else:
            abs_pos_h = self.rel_emb_h[self.rel_idx_h.view(-1)]
            abs_pos_w = self.rel_emb_w[self.rel_idx_w.view(-1)]
            abs_pos_h = rearrange(abs_pos_h, "(q k) c -> q k c", q=self.h)  # [qh, kh, c]
            abs_pos_w = rearrange(abs_pos_w, "(q k) c -> q k c", q=self.w)  # [qw, kw, c]

        if rois is not None:
            #  rois: [B, (H, W), D],
            rois = rearrange(rois, "b (h w) d -> b h w d", h=self.h)
            if self.norm_roi:  # weights within rois should be <= 1
                rois = F.softmax(rois, dim=-1)
            # abs_pos_h = abs_pos_h[:, :, None].repeat(1, 1, kw, 1)  # [qh, kh, kw, c]
            # abs_pos_w = abs_pos_w[:, None].repeat(1, kh, 1, 1)  # [qw, kh, kw, c]

            abs_pos_h = torch.einsum("b h d, q h c -> b q d c", rois.sum(2), abs_pos_h)
            abs_pos_w = torch.einsum("b w d, q w c -> b q d c", rois.sum(1), abs_pos_w)
            # abs_pos_w = abs_pos_w.sum(2)

            q = rearrange(q, "b (qh qw) g (t c) -> b qh qw g t c", qh=self.h, t=2)
            logits_h = torch.einsum("b h w g c, b h k c -> b g h w k", q[..., 0, :], abs_pos_h)
            logits_w = torch.einsum("b h w g c, b w k c -> b g h w k", q[..., 1, :], abs_pos_w)

            attn = rearrange(attn, "b g (h w) k -> b g h w k", h=self.h)
            attn += logits_h
            attn += logits_w
            return rearrange(attn, "b g h w k -> b g (h w) k")

        # if self.training and self.drop_ratio:
        #     abs_pos_h *= torch.rand(self.h, self.h, 1, device=q.device) > self.drop_ratio
        #     abs_pos_w *= torch.rand(self.w, self.w, 1, device=q.device) > self.drop_ratio

        # abs_pos_h = abs_pos_h[:, None, :, None]
        # abs_pos_w = abs_pos_w[None, :, None, :]
        #
        # abs_pos_h = abs_pos_h.repeat(1, self.w, 1, self.w, 1)
        # abs_pos_w = abs_pos_w.repeat(self.h, 1, self.h, 1, 1)
        # abs_pos = torch.cat([abs_pos_h, abs_pos_w], dim=-1)
        # abs_pos = rearrange(abs_pos, "qh qw kh kw c -> (qh qw) (kh kw) c")
        #
        # logits = torch.einsum("b q p g c, q k c -> b g q k p", q, abs_pos)
        # attn += logits
        # return attn

        q = rearrange(q, "b (qh qw) p g (n c) -> b qh qw p g n c", qh=self.h, n=2)
        logits_h = torch.einsum("b h w p g c, h k c -> b g h w k p", q[..., 0, :], abs_pos_h)
        logits_w = torch.einsum("b h w p g c, w k c -> b g h w k p", q[..., 1, :], abs_pos_w)
        logits_h = rearrange(logits_h, "b g h w k p -> b g (h w) k 1 p")
        logits_w = rearrange(logits_w, "b g h w k p -> b g (h w) 1 k p")

        attn = rearrange(attn, "b g q (kh kw) p -> b g q kh kw p", kh=self.h, kw=self.w)
        attn += logits_h
        attn += logits_w
        return rearrange(attn, "b g q h w p -> b g q (h w) p")


class SwinRelPos(nn.Module):
    def __init__(self, h, w, num_heads=1):
        super(SwinRelPos, self).__init__()
        self.h, self.w = h, w
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * h - 1) * (2 * w - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += h - 1  # shift to start from 0
        relative_coords[:, :, 1] += w - 1
        relative_coords[:, :, 0] *= 2 * w - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, attn):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.h * self.w, self.h * self.w, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias[None, ..., None]


class RelPos(nn.Module):
    """Relative Position Self Attention"""

    def __init__(self, k_len, q_len, dim, win_len=1, learn_map=False):
        super(RelPos, self).__init__()
        rel_length = 2 * win_len  # - 1
        self.rel_emb_w = nn.Parameter(torch.Tensor(rel_length, dim // 2))
        self.rel_emb_h = nn.Parameter(torch.Tensor(rel_length, dim // 2))
        onehot = torch.zeros(q_len, k_len, rel_length)
        self.q_len, self.k_len = q_len, k_len
        rel_idx = torch.arange(-win_len, win_len)  # win_len+1

        for i in range(q_len):
            for j in range(k_len):
                for r in range(rel_length):
                    if rel_idx[r] == j - i:  # *stride
                        onehot[i, j, r] = 1

        if learn_map:
            self.map = nn.Parameter(onehot, requires_grad=True)
        else:
            self.map = onehot
        nn.init.normal_(self.rel_emb_w, std=dim ** -0.5)
        nn.init.normal_(self.rel_emb_h, std=dim ** -0.5)
        trunc_normal_(self.rel_emb_w, std=.02)
        trunc_normal_(self.rel_emb_h, std=.02)

    def forward(self, q):
        onehot = self.map.to(device=q.device)

        abs_pos_h = torch.einsum("q k r, r c -> q k c", onehot, self.rel_emb_h)
        abs_pos_h = abs_pos_h[:, None, :, None]
        abs_pos_w = torch.einsum("q k r, r c -> q k c", onehot, self.rel_emb_w)
        abs_pos_w = abs_pos_w[None, :, None, :]

        abs_pos_h = abs_pos_h.repeat(1, self.q_len, 1, self.k_len, 1)
        abs_pos_w = abs_pos_w.repeat(self.q_len, 1, self.k_len, 1, 1)

        pos = torch.cat([abs_pos_h, abs_pos_w], dim=-1)
        pos = rearrange(pos, "qh qw kh kw c -> (qh qw) (kh kw) c")
        logits = torch.einsum("b q p g c, q k c -> b g q k p", q, pos)
        return logits


class DPSConvPos(nn.Module):
    def __init__(self, dim, k):
        super(DPSConvPos, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=k, padding=k//2, groups=dim, bias=False)

    def forward(self, v, H):
        v = rearrange(v, "b (h w) c -> b c h w", h=H)
        v = self.conv(v)
        return rearrange(v, "b c h w -> b (h w) c")


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, H=0, W=0):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        if H and W:
            mask = torch.zeros((1, H, W), dtype=torch.bool)
            self.pos = self.make_pos(mask)
        else:
            self.pos = None

    def make_pos(self, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)  # .permute(0, 3, 1, 2)
        return rearrange(pos, "b h w c -> b (h w) c")

    def forward(self, x, H=0, mask=None):
        B = x.shape[0]
        if self.pos is None:
            W = int(x.shape[1] / H)
            mask = torch.zeros((B, H, W), device=x.device, dtype=torch.bool) if mask is None else mask
            mask = mask.view(B, H, W)
            return self.make_pos(mask)
        else:
            return self.pos.to(x.device).expand(B, -1, -1)
