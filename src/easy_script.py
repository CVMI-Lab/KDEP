import torch
import clip
from PIL import Image
from collections import OrderedDict

device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
model, preprocess = clip.load("RN50", device=device)

state_dict=model.state_dict()
new_ckpt = OrderedDict()
for k, v in state_dict.items():
    if 'visual' in k:
        new_ckpt[k.replace('visual.', '')] = v

torch.save(new_ckpt, '/opt/tiger/filter_transfer/pretrained-models/RN50.clean')
import ipdb
ipdb.set_trace(context=20)
a=1