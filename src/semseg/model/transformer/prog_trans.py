import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange
from .misc import DropPath, trunc_normal_
from .pos_embed import RelPos, SwinRelPos, FullRelPos, DPSConvPos, PositionEmbeddingSine


Norm = nn.LayerNorm


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features) or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SimpleReasoning(nn.Module):
    def __init__(self, np, dim, act=None):
        super(SimpleReasoning, self).__init__()
        self.norm = Norm(dim)
        self.linear = nn.Conv1d(np, np, kernel_size=1, bias=False)
        self.act = act() if act is not None else nn.Identity()

    def forward(self, x):
        tokens = self.norm(x)
        tokens = self.linear(tokens)
        tokens = self.act(tokens)
        return x + tokens


class AnyAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 dim_qk=0,
                 dim_v=0,
                 merge_qkv=False,
                 merge_qk=False,
                 merge_kv=False,
                 has_to_q=True,
                 has_to_k=True,
                 has_to_v=True,
                 has_proj=True,
                 self_qpos_type=None,
                 self_kpos_type=None,
                 self_dps_pos_k=0,
                 num_samples=0,
                 dps_pos_apply_on="x",
                 update_pos=False):
        super(AnyAttention, self).__init__()
        dim_per_head = dim_qk // num_heads if dim_qk else dim // num_heads
        has_to_kv = has_to_k and has_to_v
        has_to_qk = has_to_k and has_to_q
        if merge_qkv:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        elif merge_qk:
            self.to_qk = nn.Linear(dim, dim * 2, bias=False) if has_to_qk else nn.Identity()
            self.to_v = nn.Linear(dim, dim, bias=False) if has_to_v else nn.Identity()
        elif merge_kv:
            self.to_kv = nn.Linear(dim, dim * 2, bias=False) if has_to_kv else nn.Identity()
            self.to_q = nn.Linear(dim, dim, bias=False) if has_to_q else nn.Identity()
        else:
            self.to_q = nn.Linear(dim, dim, bias=False) if has_to_q else nn.Identity()
            self.to_k = nn.Linear(dim, dim, bias=False) if has_to_k else nn.Identity()
            self.to_v = nn.Linear(dim, dim, bias=False) if has_to_v else nn.Identity()

        if self_qpos_type == "abs":
            assert num_samples > 0
            self.self_qpos = nn.Parameter(torch.zeros(1, num_samples, 1, dim_per_head))
            trunc_normal_(self.self_qpos, std=0.02)
        elif self_qpos_type == "rel":
            assert num_samples > 0
            self.self_qpos = RelPos(num_samples, num_samples, dim_per_head)
            trunc_normal_(self.self_qpos, std=0.02)
        else:
            self.self_qpos = None

        if self_kpos_type == "abs":
            assert num_samples > 0
            self.self_kpos = nn.Parameter(torch.zeros(1, num_samples, 1, dim_per_head))
            trunc_normal_(self.self_kpos, std=0.02)
        elif self_kpos_type == "rel":
            assert num_samples > 0
            self.self_kpos = RelPos(num_samples, num_samples, dim_per_head)
            trunc_normal_(self.self_kpos, std=0.02)
        else:
            self.self_kpos = None

        self.dps_pos = DPSConvPos(dim, k=self_dps_pos_k) if self_dps_pos_k else None
        self.scale = dim_per_head ** (-0.5)
        self.num_heads = num_heads
        self.update_pos = update_pos
        self.merge_qkv = merge_qkv
        self.merge_qk = merge_qk
        self.merge_kv = merge_kv
        self.dps_pos_apply_on = dps_pos_apply_on

        dim_out = dim_v if dim_v else dim
        self.proj = nn.Linear(dim_out, dim_out) if has_proj else nn.Identity()

    def split_qkv(self, tensor, to_func, t=3):
        if not isinstance(to_func, nn.Identity):
            temp = rearrange(tensor, "b n (t c) -> b n t c", t=t)
            tensor_list = [temp[:, :, i] for i in range(t)]
        else:
            tensor_list = [tensor for _ in range(t)]
        return tuple(tensor_list)

    def get_qkv(self, x, q, k, v):
        if self.merge_qkv:
            qkv = self.to_qkv(x)
            q, k, v = self.split_qkv(qkv, self.to_qkv, t=3)
        elif self.merge_qk:
            qk = self.to_qk(x)
            q, k = self.split_qkv(qk, self.to_qk, t=2)
            v = self.to_v(v)
        elif self.merge_kv:
            q = self.to_q(q)
            kv = self.to_kv(x)
            k, v = self.split_qkv(kv, self.to_kv, t=2)
        else:
            q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        return q, k, v

    def apply_abs_pos(self, t, p, kind="q", H=0):
        def apply_pos(tensor, pos):
            if isinstance(pos, PositionEmbeddingSine):
                return t + pos(tensor, H)
            elif len(pos.shape) == len(tensor.shape):
                return tensor + pos
            else:
                tensor = rearrange(tensor, "b n (g c) -> b n g c", g=self.num_heads)
                tensor = tensor + pos
                return rearrange(tensor, "b n g c -> b n (g c)")

        self_pos = getattr(self, f"self_{kind}pos")
        if self_pos is not None:
            return apply_pos(t, self_pos)
        elif p is not None:
            return apply_pos(t, p)
        else:
            return t

    def forward(self, x=None, q=None, k=None, v=None, qpos=None, kpos=None, H=0, W=0, mask=None):
        x = self.dps_pos(x, H) if self.dps_pos is not None and self.dps_pos_apply_on == "x" else x
        q = self.apply_abs_pos(q if q is not None else x, qpos, kind="q", H=H)
        k = self.apply_abs_pos(k if k is not None else x, kpos, kind="k", H=H)
        v = v if v is not None else x

        q, k, v = self.get_qkv(x, q, k, v)
        v = self.dps_pos(v, H) if self.dps_pos is not None and self.dps_pos_apply_on == "v" else v

        # reshape
        q = rearrange(q, "b n (g c) -> b n g c", g=self.num_heads)
        k = rearrange(k, "b n (g c) -> b n g c", g=self.num_heads)
        v = rearrange(v, "b n (g c) -> b n g c", g=self.num_heads)

        # calculation
        attn = torch.einsum("b q g c, b k g c -> b q g k", q, k)
        if isinstance(self.self_kpos, SwinRelPos):
            attn += self.self_kpos(attn)
        elif isinstance(self.self_kpos, FullRelPos):
            attn = self.self_kpos(q, attn)
        elif isinstance(self.self_kpos, RelPos):
            attn += self.self_kpos(q)
        attn *= self.scale
        attn = attn.masked_fill(mask.bool(), value=float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = attn.masked_fill(mask.bool(), value=0)
        out = torch.einsum("b q g k, b k g c -> b q g c", attn, v)
        out = rearrange(out, "b q g c -> b q (g c)")
        out = self.proj(out)
        if self.update_pos:
            kpos = kpos.expand(out.shape[0], -1, -1) if kpos.shape[0] == 1 else kpos
            qpos = rearrange(kpos, "b n (g c) -> b n g c", g=self.num_heads)
            qpos = torch.einsum("b n g c, b n g d -> b c g d", attn, qpos)
            return out, qpos
        return out


class SoftRPN(nn.Module):
    def __init__(self,
                 *,
                 dim,
                 num_points=196,
                 has_rpn_kv=True,
                 has_rpn_q=True,
                 has_rpn_proj=True,
                 has_comm=False,
                 comm_ratio=1,
                 num_rpn_heads=1,
                 drop_ratio=0.1,
                 self_attn_pos=None,
                 rpn_self_pos=None,
                 rpn_dps_pos_k=0,
                 update_pos=False,
                 token_reason=True,
                 **kwargs):
        super(SoftRPN, self).__init__()
        self.norm1 = Norm(dim) if has_comm else None
        self.norm2 = Norm(dim)

        self.self_attn = AnyAttention(dim,
                                      num_rpn_heads,
                                      dim_qk=dim // comm_ratio if comm_ratio > 1 else 0,
                                      merge_qkv=True,
                                      self_qpos_type=self_attn_pos,
                                      self_kpos_type=self_attn_pos,
                                      num_samples=num_points) if has_comm else None
        self.rpn_attn = AnyAttention(dim,
                                     num_rpn_heads,
                                     merge_kv=True,
                                     has_to_q=has_rpn_q,
                                     has_to_k=has_rpn_kv,
                                     has_to_v=has_rpn_kv,
                                     has_proj=has_rpn_proj,
                                     self_qpos_type=rpn_self_pos,
                                     update_pos=update_pos,
                                     self_dps_pos_k=rpn_dps_pos_k,
                                     num_samples=num_points)
        self.update_pos = update_pos
        self.drop_path = DropPath(drop_prob=drop_ratio) if drop_ratio else nn.Identity()
        self.reason = SimpleReasoning(num_points, dim) if token_reason else nn.Identity()

    def forward(self, enc_feat, rpn_tokens=None, rpn_pos=0., spa_pos=None, H=0, mask=None):
        if self.self_attn is not None:
            rpn_tokens = rpn_tokens + self.drop_path(
                self.self_attn(
                    x=self.norm1(rpn_tokens),
                    qpos=rpn_pos,
                    kpos=rpn_pos)
            )

        rpn_out = self.norm2(rpn_tokens)
        if self.update_pos:
            rpn_out, rpn_pos = self.rpn_attn(
                x=enc_feat,
                q=rpn_out,
                qpos=rpn_pos,
                kpos=spa_pos,
                H=H,
                mask=mask
            )
        else:
            rpn_out = self.rpn_attn(
                x=enc_feat,
                q=rpn_out,
                qpos=rpn_pos,
                kpos=spa_pos,
                H=H,
                mask=mask
            )

        rpn_tokens = rpn_tokens + self.drop_path(rpn_out)
        rpn_tokens = self.reason(rpn_tokens)

        return rpn_tokens, rpn_pos


class PatchEmbed(nn.Module):
    def __init__(self,
                 patch_size,
                 use_pool=False,
                 has_proj=False,
                 proj_bias=False,
                 in_ch=0,
                 out_ch=0,
                 return_hw=False,
                 dps_conv_k=0,
                 window_size=0):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.return_hw = return_hw
        self.window_size = window_size
        if patch_size == 1:
            self.to_token = nn.Identity()
        elif dps_conv_k:
            self.to_token = nn.Conv2d(in_ch,
                                      in_ch,
                                      kernel_size=dps_conv_k,
                                      padding=dps_conv_k // 2,
                                      stride=patch_size,
                                      groups=in_ch)
        elif use_pool:
            self.to_token = nn.AvgPool3d(kernel_size=(patch_size, patch_size, 1),
                                         stride=(patch_size, patch_size, 1))
        else:
            self.to_token = nn.Conv3d(1, 1,
                                      kernel_size=(patch_size, patch_size, 1),
                                      stride=(patch_size, patch_size, 1))

        self.proj = nn.Linear(in_ch, out_ch, bias=proj_bias) if has_proj else nn.Identity()

    def pad_tensor(self, x, mask, H, W):
        if mask is None:
            mask = x.new_zeros((1, 1, H, W))
        H_mask, W_mask = mask.shape[-2:]
        if H_mask != H or W_mask != W:
            mask = F.interpolate(mask, (H, W), mode='nearest')

        pad_l = pad_t = 0
        pad_r = int(math.ceil(W / self.window_size)) * self.window_size - W
        pad_b = int(math.ceil(H / self.window_size)) * self.window_size - H
        # x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        mask = F.pad(mask, (pad_l, pad_r, pad_t, pad_b), value=1)
        return x, mask, H+pad_b, W+pad_r

    def forward(self, x, H=0, W=0, mask=None):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        out = self.to_token(x.contiguous())
        H, W = out.shape[2], out.shape[3]

        out, mask, H, W = self.pad_tensor(out, mask, H, W)
        out = rearrange(out, "b c h w -> b (h w) c")
        out = self.proj(out.contiguous())
        # out = rearrange(out, "b c h w -> b (h w) c")
        return out, H, W, mask


class GlobalAttn(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 pool_proj=True,
                 sr_ratio=1,
                 num_points=0,
                 prev_np=0,
                 has_router=False,
                 use_pool_global=False,
                 pos=None,
                 dim_qk=0,
                 has_cls_token=False,
                 rpn_dim_q=0,
                 rpn_ffn=True,
                 act=nn.GELU,
                 **ret_args):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.has_cls_token = has_cls_token
        self.pos = pos
        if num_points:
            self.decoder = SoftRPN(dim=dim,
                                   num_points=num_points,
                                   prev_np=prev_np,
                                   has_router=has_router,
                                   num_heads=num_heads,
                                   dim_q=rpn_dim_q,
                                   **ret_args)
        else:
            self.decoder = PatchEmbed(sr_ratio, has_proj=pool_proj, use_pool=use_pool_global, in_ch=dim, out_ch=dim)

        self.attn = AnyAttention(dim,
                                 num_heads,
                                 dim_qk=dim_qk,
                                 merge_kv=True)
        self.norm1 = Norm(dim)
        self.norm2 = Norm(dim)
        self.rpn_ffn = Mlp(dim, hidden_features=dim, act_layer=act) if rpn_ffn else None

    def forward(self, x, H=0, rpn_tokens=None, rpn_pos=0., spa_pos=None, mask=None):
        """
            x: [B, N, C]
        """
        dec_mask = rearrange(mask.squeeze(1), "b h w -> b 1 1 (h w)")
        enc_feat = x[:, 1:] if self.has_cls_token else x
        # enc_feat = enc_feat + spa_pos if spa_pos is not None else enc_feat
        if isinstance(self.decoder, SoftRPN):
            rpn_out, rpn_pos = self.decoder(enc_feat,
                                            rpn_tokens=rpn_tokens,
                                            rpn_pos=rpn_pos,
                                            spa_pos=spa_pos,
                                            H=H,
                                            mask=dec_mask)
        else:
            rpn_out = self.decoder(enc_feat, H=H)

        if self.rpn_ffn is not None:  # add adtivation
            rpn_out = rpn_out + self.rpn_ffn(self.norm1(rpn_out))

        enc_mask = rearrange(mask.squeeze(1), "b h w -> b (h w) 1 1")
        normed_tokens = self.norm2(rpn_out)
        enc_feat = self.attn(q=enc_feat,
                             x=normed_tokens,
                             qpos=spa_pos,
                             kpos=rpn_pos,
                             H=H,
                             mask=enc_mask)

        rpn_tokens = None if rpn_tokens is None else rpn_out
        return enc_feat, rpn_tokens, rpn_pos


class LocalAttn(nn.Module):
    def __init__(self,
                 out_ch,
                 patch_size=7,
                 overlap=3,
                 ratio=1,
                 num_heads=1,
                 dim_qk=0,
                 stride_kv=1,
                 pool_type="avg",
                 act=nn.GELU,
                 no_proj=False,
                 rel_pos=None,
                 qkv_bias=False,
                 **ret_args):
        super(LocalAttn, self).__init__()
        # patch_size_idt = math.ceil(patch_size / stride) if stride > 1 else patch_size
        patch_size_idt = patch_size
        hdim = int(out_ch // ratio)
        self.act = act

        if stride_kv > 1:
            if pool_type == "avg":
                self.pool_kv = nn.AvgPool2d(kernel_size=stride_kv, stride=stride_kv)
            elif pool_type == "max":
                self.pool_kv = nn.MaxPool2d(kernel_size=stride_kv, stride=stride_kv)
        else:
            self.pool_kv = nn.Identity()

        if dim_qk:
            self.to_qk = nn.Linear(hdim, dim_qk * 2, bias=qkv_bias)
            self.to_v = nn.Linear(hdim, hdim, bias=qkv_bias)
            self.to_qkv = None
            self.scale = dim_qk ** -0.5
        else:
            self.to_qkv = nn.Linear(hdim, hdim * 3, bias=qkv_bias)
            self.scale = (hdim // num_heads) ** -0.5

        self.proj_attn = nn.Identity() if no_proj else nn.Linear(hdim, hdim)

        self.rel_pos = rel_pos
        self.overlap = overlap
        self.patch_size = patch_size // stride_kv
        self.patch_size_idt = patch_size_idt
        self.hdim = hdim
        self.ratio = ratio
        self.num_heads = num_heads
        # self.drop_path = DropPath(drop_ratio) if drop_ratio else nn.Identity()

    def cal_pad(self, l, k, o, step=0):
        step = step if step else int(l // k)
        diff = l - 2 * o - step * k
        return diff if diff >= 0 else -diff

    def to_patch(self, x, patch_size, H, W, mask):
        """
        :param x: [n, H * W, qkv_dim]
        :param patch_size: default-7
        :param mask: [1, 1, H, W] pad by 1
        :return x: [n, patch_size_h * patch_size_w, num_patch_h * num_patch_w, c]
        :return mask: [n, 1, patch_size_h * patch_size_w, num_patch_h * num_patch_w]
        """
        x = rearrange(x, "b (h w) c -> b h w c", h=H, w=W)
        pad_l = pad_t = 0
        pad_r = int(math.ceil(W / self.patch_size)) * self.patch_size - W
        pad_b = int(math.ceil(H / self.patch_size)) * self.patch_size - H
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        mask = F.pad(mask, (pad_l, pad_r, pad_t, pad_b), value=1)

        x = rearrange(x, "b (sh kh) (sw kw) c -> b (kh kw) (sh sw) c", kh=patch_size, kw=patch_size)
        mask = rearrange(mask, "b c (sh kh) (sw kw) -> b c (kh kw) (sh sw)", kh=patch_size, kw=patch_size)

        return x, mask, H+pad_b, W+pad_r

    def forward(self, x, H, W, mask=None):
        qkv = self.to_qkv(x)  # quicker
        ori_H, ori_W = H, W
        qkv, k_mask, H, W = self.to_patch(qkv, self.patch_size, H, W, mask)
        qkv = rearrange(qkv, "b k p (n g c) -> b k p n g c", n=3, g=self.num_heads)
        q, kv = qkv[:, :, :, 0], self.pool_kv(qkv[:, :, :, 1:])
        k, v = kv[:, :, :, 0], kv[:, :, :, 1]

        attn = torch.einsum("b q p g c, b k p g c -> b g q k p", q, k)

        if isinstance(self.rel_pos, SwinRelPos):
            attn += self.rel_pos(attn)
        elif isinstance(self.rel_pos, FullRelPos):
            attn = self.rel_pos(q, attn)
        else:
            attn += self.rel_pos(q)
        attn *= self.scale

        attn = attn.masked_fill(k_mask[:, None].bool(), value=float('-inf'))
        attn = F.softmax(attn, dim=-2)
        attn = attn.masked_fill(k_mask[:, None].bool(), value=0)

        out = torch.einsum("b g q k p, b k p g c -> b q p g c", attn, v)
        out = rearrange(out, "b q p g c -> b q p (g c)")

        out = rearrange(out, "b (kh kw) (sh sw) c -> b (sh kh) (sw kw) c",
                        kh=self.patch_size, sh=H // self.patch_size)
        out = out[:, :ori_H, :ori_W, :]
        out = rearrange(out, "b h w c -> b (h w) c")
        out = self.proj_attn(out)
        # out = out[:, :, q_pad_h // 2:q_pad_h // 2 + H, q_pad_w // 2: q_pad_w // 2 + W]

        # out = idt + self.drop_path(out)
        # out = out + self.drop_path(self.ffn(self.norm_ffn(out)))

        return out


class SeparableAttn(nn.Module):
    def __init__(self,
                 in_ch,
                 dim,
                 ffn_exp=4,
                 act=nn.GELU,
                 drop_path=0.1,
                 rel_pos=None,
                 abs_pos=None,
                 prev_np=0,
                 head_dim_qk=0,
                 head_dim_v=0,
                 num_heads=1,
                 num_points=0,
                 has_local=True,
                 has_ffn1=True,
                 has_ffn2=True,
                 act_attn1=False,
                 act_attn2=False,
                 stride=1,
                 use_pool=True,
                 pool_proj=True,
                 dps_conv_k=0,
                 patch_size=0,
                 **kwargs):
        super(SeparableAttn, self).__init__()
        dim = head_dim_v * num_heads if head_dim_v else dim
        head_dim_v = head_dim_v if head_dim_v else dim // num_heads
        dim_qk = head_dim_qk * num_heads if head_dim_qk and head_dim_qk != head_dim_v else 0
        if in_ch != dim:
            self.proj = PatchEmbed(stride,
                                   use_pool=use_pool,
                                   in_ch=in_ch,
                                   out_ch=dim,
                                   has_proj=pool_proj,
                                   return_hw=True,
                                   window_size=patch_size,
                                   dps_conv_k=dps_conv_k)
            self.proj_token = nn.Sequential(
                nn.Conv1d(prev_np, num_points, kernel_size=1, bias=False) if prev_np != num_points else nn.Identity(),
                nn.Linear(in_ch, dim),
                Norm(dim)
            )
            self.proj_norm = Norm(dim)
        else:
            self.proj, self.proj_token, self.proj_norm = None, None, None

        if has_local:
            self.local_attn = LocalAttn(dim,
                                        rel_pos=rel_pos,
                                        drop_ratio=drop_path,
                                        num_heads=num_heads,
                                        dim_qk=dim_qk,
                                        patch_size=patch_size,
                                        act=act, **kwargs)
            self.norm1 = Norm(dim)
            self.norm_ffn1 = Norm(dim) if has_ffn1 else None
            self.ffn1 = Mlp(dim, hidden_features=dim * ffn_exp, act_layer=act) if has_ffn1 else None
            self.act1 = act() if act_attn1 else nn.Identity()
        else:
            self.local_attn = None

        self.norm2 = Norm(dim)
        self.abs_pos = abs_pos
        self.global_attn = GlobalAttn(dim,
                                      prev_np=prev_np,
                                      pos=abs_pos,
                                      num_heads=num_heads,
                                      dim_qk=dim_qk,
                                      num_points=num_points,
                                      **kwargs)
        self.norm_ffn2 = Norm(dim) if has_ffn2 else None
        self.ffn2 = Mlp(dim, hidden_features=dim * ffn_exp, act_layer=act) if has_ffn2 else None
        self.act2 = act() if act_attn2 else nn.Identity()
        self.drop_path = DropPath(drop_path)

    def forward(self, x, H, W, cls_token=None, rpn_tokens=None, rpn_pos=0., mask=None):
        if self.proj is not None:
            out, H, W, mask = self.proj(x, H=H, W=W, mask=mask)
            out = self.proj_norm(out)
        else:
            out = x

        if self.proj_token is not None:
            rpn_tokens = self.proj_token(rpn_tokens)

        if self.local_attn is not None:
            out = out + self.drop_path(self.act1(self.local_attn(self.norm1(out), H, W, mask)))
            if self.ffn1 is not None:
                out = out + self.drop_path(self.ffn1(self.norm_ffn1(out)))

        # if isinstance(self.abs_pos, nn.Parameter):
        #     out += self.abs_pos
        if cls_token is not None:
            out = torch.cat([cls_token, out], dim=1)

        global_out, rpn_tokens, rpn_pos = self.global_attn(self.norm2(out),
                                                           H=H,
                                                           rpn_tokens=rpn_tokens,
                                                           rpn_pos=rpn_pos,
                                                           spa_pos=self.abs_pos,
                                                           mask=mask)
        out = out + self.drop_path(self.act2(global_out))
        if self.ffn2 is not None:
            out = out + self.drop_path(self.ffn2(self.norm_ffn2(out)))
        # out = rearrange(out, "b (h w) c -> b c h w", h=H)
        if cls_token is not None:
            cls_token, out = out[:, :1], out[:, 1:]
        return out, H, W, cls_token, rpn_tokens, rpn_pos, mask


class Stage(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 num_blocks,
                 patch_size=7,
                 overlap=0,
                 num_heads=1,
                 stride=1,
                 num_points=0,
                 last_np=0,
                 H=56,
                 W=56,
                 pos_type="rel",
                 rpn_pos_type="abs",
                 win_len=1,
                 head_dim_qk=0,
                 head_dim_v=0,
                 has_cls_token=False,
                 is_prog_rpn=True,
                 **kwargs):
        super(Stage, self).__init__()
        if has_cls_token and in_ch != out_ch and stride > 1:
            self.cls_fc = nn.Linear(in_ch, out_ch)
        else:
            self.cls_fc = nn.Identity()

        if pos_type == "swin":
            self.rel_pos = SwinRelPos(patch_size, patch_size, num_heads)
        elif pos_type == "rel":
            self.rel_pos = RelPos(patch_size + 2 * overlap,
                                  patch_size,
                                  head_dim_qk if head_dim_qk else out_ch // num_heads,
                                  win_len=win_len)
        else:
            self.rel_pos = FullRelPos(patch_size + 2 * overlap,
                                      patch_size,
                                      head_dim_qk if head_dim_qk else out_ch // num_heads)

        if rpn_pos_type == "abs":
            self.abs_pos = nn.Parameter(torch.zeros(1, H * W,
                                                    head_dim_v * num_heads if head_dim_v else out_ch))
            trunc_normal_(self.abs_pos, std=.02)
        elif rpn_pos_type == "dps":
            self.abs_pos = None
        elif rpn_pos_type == "dpsk":
            self.abs_pos = nn.Parameter(torch.zeros(1, H * W,
                                                    head_dim_v * num_heads if head_dim_v else out_ch))
            trunc_normal_(self.abs_pos, std=.02)
        elif rpn_pos_type == "sin":
            self.abs_pos = PositionEmbeddingSine(out_ch // 2, normalize=True)
        else:
            self.abs_pos = FullRelPos(H, W, dim=head_dim_qk if head_dim_qk else out_ch // num_heads)

        self.rpn_pos = nn.Parameter(torch.Tensor(1, num_points, 1, out_ch // num_heads)) if is_prog_rpn else 0
        if is_prog_rpn:
            init.kaiming_uniform_(self.rpn_pos, a=math.sqrt(5))
            trunc_normal_(self.rpn_pos, std=.02)
        blocks = [
            SeparableAttn(
                in_ch,
                out_ch,
                patch_size=patch_size,
                overlap=overlap,
                num_heads=num_heads,
                stride=stride,
                rel_pos=self.rel_pos,
                abs_pos=self.abs_pos,
                num_points=num_points,
                prev_np=last_np,
                head_dim_qk=head_dim_qk,
                head_dim_v=head_dim_v,
                has_cls_token=has_cls_token,
                is_prog_rpn=is_prog_rpn,
                **kwargs
            )]
        [
            blocks.append(
                SeparableAttn(
                    out_ch,
                    out_ch,
                    patch_size=patch_size,
                    overlap=overlap,
                    num_heads=num_heads,
                    stride=1,
                    rel_pos=self.rel_pos,
                    abs_pos=self.abs_pos,
                    num_points=num_points,
                    prev_np=num_points,
                    head_dim_qk=head_dim_qk,
                    head_dim_v=head_dim_v,
                    has_cls_token=has_cls_token,
                    is_prog_rpn=is_prog_rpn,
                    **kwargs)
            ) for _ in range(num_blocks - 1)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, H, W, cls_token=None, rpn_tokens=None, mask=None):
        if cls_token is not None:
            cls_token = self.cls_fc(cls_token)
        rpn_pos = self.rpn_pos
        rpn_pos = rpn_pos.expand(x.shape[0], -1, -1, -1)
        for blk in self.blocks:
            x, H, W, cls_token, rpn_tokens, rpn_pos, mask = blk(x,
                                                                  H,
                                                                  W,
                                                                  cls_token,
                                                                  rpn_tokens,
                                                                  rpn_pos,
                                                                  mask)
        return x, H, W, cls_token, rpn_tokens, mask


class ProgressiveTransformer(nn.Module):
    def __init__(self,
                 inplanes=64,
                 num_layers=(3, 4, 6, 3),
                 num_chs=(256, 512, 1024, 2048),
                 num_strides=(1, 2, 2, 2),
                 input_size=224,
                 num_cls=1000,
                 block=SeparableAttn,
                 is_patch_attn=(False, False, False, False),
                 pre_act_attn=(False, False, False, False),
                 has_ffn=(True, True, True, True),
                 global_attn=(False, False, False, False),
                 no_proj=(False, False, False, False),
                 head_dim_qk=(0, 0, 0, 0),
                 head_dim_v=(0, 0, 0, 0),
                 num_heads=(1, 1, 1, 1),
                 num_points=(1, 1, 1, 1),
                 patch_sizes=(1, 1, 1, 1),
                 sr_ratio=(1, 1, 1, 1),
                 num_rpn_heads=(1, 1, 1, 1),
                 act=nn.GELU,
                 no_pos_wd=False,
                 has_cls_token=False,
                 is_prog_rpn=True,
                 has_last_decoder=False,
                 BN=nn.BatchNorm2d,
                 **blkargs):
        super(ProgressiveTransformer, self).__init__()
        self.block = block
        self.act = act()
        self.depth = len(num_layers)
        self.no_pos_wd = no_pos_wd

        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, padding=3, stride=2, bias=False)
        self.norm1 = BN(inplanes)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.rpn_tokens = nn.Parameter(torch.Tensor(1, num_points[0], inplanes)) if is_prog_rpn else None

        num_chs = [dim * num_heads[sid] if dim else num_chs[sid] for sid, dim in enumerate(head_dim_v)]

        last_chs = [inplanes, *num_chs[:-1]]
        last_nps = [num_points[0], *num_points[:-1]]
        H = [input_size // 4, input_size // 8, input_size // 8, input_size // 8]
        W = [input_size // 4, input_size // 8, input_size // 8, input_size // 8]

        for i, n_l in enumerate(num_layers):
            setattr(self,
                    "layer_{}".format(i),
                    Stage(last_chs[i],
                          num_chs[i],
                          n_l,
                          stride=num_strides[i],
                          is_patch_attn=is_patch_attn[i],
                          num_heads=num_heads[i],
                          num_rpn_heads=num_rpn_heads[i],
                          has_ffn=has_ffn[i],
                          patch_size=patch_sizes[i],
                          global_attn=global_attn[i],
                          no_proj=no_proj[i],
                          pre_act=pre_act_attn[i],
                          act=act,
                          num_points=num_points[i],
                          H=H[i],
                          W=W[i],
                          sr_ratio=sr_ratio[i],
                          last_np=last_nps[i],
                          head_dim_qk=head_dim_qk[i],
                          head_dim_v=head_dim_v[i],
                          has_cls_token=has_cls_token,
                          is_prog_rpn=is_prog_rpn,
                          **blkargs
                          )
                    )

        if has_cls_token or has_last_decoder:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, num_chs[0]),
                                          requires_grad=True) if has_cls_token else None
            self.last_fc = nn.Linear(num_chs[-1], num_cls)
        else:
            self.last_linear = nn.Linear(num_chs[-1], num_chs[-1], bias=False)
            self.last_norm = BN(num_chs[-1])
            self.pool2 = nn.AdaptiveAvgPool2d(1)
            self.cls_token = None
            self.last_fc = nn.Conv1d(num_chs[-1], num_cls, kernel_size=1)

        if has_last_decoder:
            self.last_norm = nn.LayerNorm(num_chs[-1])
            self.last_decoder = SoftRPN(dim=num_chs[-1],
                                        num_points=num_points[-1],
                                        num_heads=num_heads[-1],
                                        rpn_self_pos="abs")
            H = W = input_size // (2 ** 5)
            self.last_pos = PositionEmbeddingSine(num_chs[-1] // 2, normalize=True)
        else:
            self.last_decoder = None
        self._init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        skip_pattern = ['rel_pos', 'abs_pos'] if self.no_pos_wd else []
        no_wd_layers = set()
        for name, param in self.named_parameters():
            for skip_name in skip_pattern:
                if skip_name in name:
                    no_wd_layers.add(name)
        if self.cls_token is not None:
            no_wd_layers.add("cls_token")
        return no_wd_layers

    def _init_weights(self):
        if self.rpn_tokens is not None:
            init.kaiming_uniform_(self.rpn_tokens, a=math.sqrt(5))
            trunc_normal_(self.rpn_tokens, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if not torch.sum(m.weight.data == 0).item() == m.num_features:  # zero gamma
                    m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            # other parameters should be initialized in its corresponding module

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        if self.pool1 is not None:
            out = self.pool1(out)

        B, _, H, W = out.shape
        out = rearrange(out, "b c h w -> b (h w) c")
        cls_token, rpn_tokens = self.cls_token, self.rpn_tokens

        if rpn_tokens is not None:
            rpn_tokens = rpn_tokens.expand(x.shape[0], -1, -1)

        for i in range(self.depth):
            layer = getattr(self, "layer_{}".format(i))
            out, H, W, cls_token, rpn_tokens = layer(out, H, W, cls_token, rpn_tokens)

        if self.last_decoder is not None:
            out, _ = self.last_decoder(self.last_norm(out),
                                       rpn_tokens,
                                       spa_pos=self.last_pos)
            out = out.mean(1)
        elif self.cls_token is not None:
            out = cls_token
        else:
            out = self.last_linear(out)
            out = self.last_norm(out.permute(0, 2, 1)[..., None])
            out = self.act(out)
            out = self.pool2(out)

        out = self.last_fc(out).squeeze()
        return out.view(out.size(0), -1)