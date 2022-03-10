from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import sys

def custom2det_R18(exp_name):
    # model.model  ->
    # orig_path = '../pretrained-models/ex99.1.t1.pt'
    orig_path = '../outdir/'+exp_name+'/checkpoint.pt.latest'
    # orig_path = '../pretrained-models/scratch-res18-random_init_clean.pt'
    orig_dict = torch.load(orig_path)
    orig_dict = orig_dict['model']
    teacher_state_dict = OrderedDict()

    for k, v in orig_dict.items():
        if ('emb' in k) or ('l2norm' in k):
            print('exception: abandom, key name is %s' % k)
        elif 'module.attacker.model.model' in k:
            teacher_state_dict[k.replace('module.attacker.model.model.', '')] = v
            # pass
        elif 'module.model.model' in k:
            teacher_state_dict[k.replace('module.model.model.', '')] = v
            pass
        else:
            print('exception:same, key name is %s' % k)
            teacher_state_dict[k] = v

    # now the ckpt is clean
    state_dict = {}

    import pickle as pkl
    # import ipdb
    # ipdb.set_trace(context=20)
    for k, v in teacher_state_dict.items():
        if 'module' in k:
            continue
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        state_dict[k] = v

    res = {"model": state_dict, "__author__": "RuifeiHe", "matching_heuristics": True}

    with open(orig_path + ".det.pkl", "wb") as f:
        pkl.dump(res, f)
    print('saving to %s' % orig_path + '.det.pkl')

def custom2det_mnv2(exp_name):
    # model.model  ->
    orig_path = '../outdir/'+exp_name+'/checkpoint.pt.latest'
    # orig_path = '../pretrained-models/scratch-res18-random_init_clean.pt'
    orig_dict = torch.load(orig_path)
    orig_dict = orig_dict['model']
    teacher_state_dict = OrderedDict()

    for k, v in orig_dict.items():
        if ('emb' in k) or ('l2norm' in k):
            print('exception: abandom, key name is %s' % k)
        elif 'module.attacker.model.model' in k:
            teacher_state_dict[k.replace('module.attacker.model.model.', '')] = v
            # pass
        elif 'module.model.model' in k:
            teacher_state_dict[k.replace('module.model.model.', '')] = v
            pass
        else:
            print('exception:same, key name is %s' % k)
            teacher_state_dict[k] = v

    # now the ckpt is clean
    state_dict = {}

    import pickle as pkl
    # import ipdb
    # ipdb.set_trace(context=20)
    for k, v in teacher_state_dict.items():
        if 'module' in k:
            continue
        # old_k = k
        # if "layer" not in k:
        #     k = "stem." + k
        # for t in [1, 2, 3, 4]:
        #     k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        # for t in [1, 2, 3]:
        #     k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        # k = k.replace("downsample.0", "shortcut")
        # k = k.replace("downsample.1", "shortcut.norm")
        # print(old_k, "->", k)
        state_dict[k] = v

    res = {"model": state_dict, "__author__": "RuifeiHe", "matching_heuristics": True}

    with open(orig_path + ".det.pkl", "wb") as f:
        pkl.dump(res, f)
    print('saving to %s' % orig_path + '.det.pkl')


if __name__ == "__main__":
    exp_name = sys.argv[1]
    arch = sys.argv[2]
    print(sys.argv[1])
    print(sys.argv[2])
    if arch == 'R18':
        custom2det_R18(exp_name)
    elif arch == 'mnv2':
        custom2det_mnv2(exp_name)
    else:
        raise ValueError
