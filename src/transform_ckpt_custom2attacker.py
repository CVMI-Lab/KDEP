from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import sys

def custom2attacker_R18(exp_name):
    # model.model  -> model
    attacker_path = '../outdir/'+exp_name+'/checkpoint.pt.latest'
    attacker_dict = torch.load(attacker_path)
    attacker_dict = attacker_dict['model']
    teacher_state_dict = OrderedDict()
    # import ipdb
    # ipdb.set_trace(context=20)
    for k, v in attacker_dict.items():
        if ('emb' in k) or ('l2norm' in k) or ('feat_scale' in k):
            print('exception: abandom, key name is %s' % k)
        elif 'module.attacker.model.model' in k:
            teacher_state_dict[k.replace('module.attacker.model.model', 'module.attacker.model')] = v
            # pass
        elif 'module.model.model' in k:
            teacher_state_dict[k.replace('module.model.model', 'module.model')] = v
            pass
        else:
            print('exception:same, key name is %s' % k)
            teacher_state_dict[k] = v
            # teacher_state_dict[k] = v

    # torch.save(teacher_state_dict, 'transformed_'+attacker_path)
    to_save = {}
    to_save['model'] = teacher_state_dict
    to_save['epoch'] = 0

    torch.save(to_save, attacker_path + '.attacker')
    print('saving to %s' % attacker_path + '.attacker')

    # torch.save(teacher_state_dict, attacker_path + '.custom')

def custom2attacker_mnv2(exp_name):
    # model.model  -> model
    attacker_path = '../outdir/'+exp_name+'/checkpoint.pt.latest'
    attacker_dict = torch.load(attacker_path)
    attacker_dict = attacker_dict['model']
    teacher_state_dict = OrderedDict()
    for k, v in attacker_dict.items():
        if ('emb' in k) or ('l2norm' in k) or ('feat_scale' in k):
            print('exception: abandom, key name is %s' % k)
        else:
            teacher_state_dict[k] = v

    to_save = {}
    to_save['model'] = teacher_state_dict
    to_save['epoch'] = 0

    torch.save(to_save, attacker_path + '.trans')
    print('saving to %s' % attacker_path + '.trans')

    # torch.save(teacher_state_dict, attacker_path + '.custom')


if __name__ == "__main__":
    exp_name = sys.argv[1]
    arch = sys.argv[2]
    print(sys.argv[1])
    print(sys.argv[2])
    if arch == 'R18':
        custom2attacker_R18(exp_name)
    elif arch == 'mnv2':
        custom2attacker_mnv2(exp_name)
    else:
        raise ValueError
