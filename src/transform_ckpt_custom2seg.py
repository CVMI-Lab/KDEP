from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import sys

def custom2seg(exp_name):
    # model.model  ->
    orig_path = '../outdir/'+exp_name+'/checkpoint.pt.latest'
    orig_dict = torch.load(orig_path)
    orig_dict = orig_dict['model']
    teacher_state_dict = OrderedDict()
    # import ipdb
    # ipdb.set_trace(context=20)
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

    torch.save(teacher_state_dict, orig_path + '.seg')
    print('saving to %s' % orig_path + '.seg')

    # torch.save(teacher_state_dict, orig_path + '.custom')


if __name__ == "__main__":
    exp_name = sys.argv[1]
    print(sys.argv[1])
    custom2seg(exp_name)

