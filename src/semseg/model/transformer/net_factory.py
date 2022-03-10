import numpy as np
from .prog_trans import ProgressiveTransformer
from timm.models.registry import register_model
from functools import partial
import torch.nn as nn
import torch
from collections import OrderedDict

__all__ = ['prog_trans_small',
           'prog_trans_medium']


@register_model
def prog_trans_small(pretrained=False, model_path=None, **cfg):
    default_params = dict(
        inplanes=64,
        num_layers=[1, 1, 3, 1],
        num_chs=[96, 192, 384, 768],
        patch_sizes=[8, 7, 7, 7],
        num_heads=[3, 6, 12, 24],
        num_strides=[1, 2, 2, 2],
        rpn_ffn=True,
        has_comm=False,
        rpn_self_pos="abs",
        rpn_pos_type="sin",
        pos_type="full",
        num_points=[64, 64, 64, 64],
        num_rpn_heads=[1, 3, 6, 12],
        ffn_exp=3,
        use_pool=False,
        has_local=True,
        dps_conv_k=7,
        token_reason=True,
        has_last_decoder=False,
        has_se=False,
        rpn_has_se=False
    )
    hyparams = {**default_params, **cfg}
    model = ProgressiveTransformer(**hyparams)

    if pretrained and model_path is not None:
        model_state_dict = torch.load(model_path)['state_dict']
        new_state_dict = OrderedDict()
        for k,v in model_state_dict.items():
            if k in ['layer_0.blocks.0.local_attn.rel_pos.rel_emb_h', 'layer_0.blocks.0.local_attn.rel_pos.rel_emb_w','layer_1.blocks.0.local_attn.rel_pos.rel_emb_h','layer_1.blocks.0.local_attn.rel_pos.rel_emb_w','layer_2.blocks.0.local_attn.rel_pos.rel_emb_h','layer_2.blocks.0.local_attn.rel_pos.rel_emb_w','layer_2.blocks.1.local_attn.rel_pos.rel_emb_h','layer_2.blocks.1.local_attn.rel_pos.rel_emb_w','layer_2.blocks.2.local_attn.rel_pos.rel_emb_h','layer_2.blocks.2.local_attn.rel_pos.rel_emb_w','layer_3.blocks.0.local_attn.rel_pos.rel_emb_h','layer_3.blocks.0.local_attn.rel_pos.rel_emb_w','last_fc.weight']:
                continue
            new_state_dict[k] = v
        # import ipdb
        # ipdb.set_trace(context=20)
        model.load_state_dict(new_state_dict, strict=False)
        print('loading from %s'%model_path)
    return model


@register_model
def prog_trans_medium(pretrained=False, model_path=None, **cfg):
    default_params = dict(
        inplanes=64,
        num_chs=(96, 192, 384, 768),
        patch_sizes=[8, 7, 7, 7],
        num_heads=[3, 6, 12, 24],
        num_points=[64, 64, 64, 128],
        num_layers=[1, 1, 8, 1],
        sr_ratio=[8, 7, 7, 7],
        is_prog_rpn=True,
        has_local=True,
        rpn_ffn=True,
        has_comm=False,
        rpn_self_pos="abs",
        rpn_pos_type="sin",
        num_rpn_heads=[1, 3, 6, 12],
        ffn_exp=3,
        use_pool=False,
        dps_conv_k=3,
        token_reason=True,
    )
    hyparams = {**default_params, **cfg}
    model = ProgressiveTransformer(**hyparams)

    if pretrained and model_path is not None:
        model.load_state_dict(model_path)
    return model