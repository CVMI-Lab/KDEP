import argparse
import os

import cox.store
import numpy as np
import torch as ch
from cox import utils
# from robustness import datasets, defaults, model_utils, train
from robustness import datasets, defaults, model_utils
import train
import train_distill
from robustness.tools import helpers
from torch import nn
from torchvision import models
import torch

from utils import constants as cs
from utils import fine_tunify, transfer_datasets


from matplotlib import pyplot as plt


def main():
    dataset = "imgnet"
    freq = np.load("outdir/freq/Imagenet_pretrained_Teacher_freq_on_{}.npy".format(dataset))
    # freq = np.load("outdir/freq/Target_data_finetuned_Teacher_freq_on_{}.npy".format(dataset))

    ax = plt.gca()
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率

    plt.rcParams['figure.figsize'] = (12.0, 4.0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    x = [i for i in range(1000)]

    plt.plot(x, freq, color='g')
    fontsize = 14
    fontweight = 'bold'
    fontproperties = {'weight': fontweight, 'size': fontsize}

    ax.set_xlabel('class id', fontsize=14, fontweight='bold')
    ax.set_ylabel('freqency', fontsize=14, fontweight='bold')

    plt.title('Imagenet pretrained Teacher freq on {} dataset'.format(dataset))
    # plt.title('Target data finetuned Teacher freq on {} dataset'.format(dataset))
    plt.savefig('outdir/freq/figures/Imagenet_pretrained_Teacher_freq_on_{}.png'.format(dataset))
    # plt.savefig('outdir/freq/figures/Target_data_finetuned_Teacher_freq_on_{}.png'.format(dataset))



if __name__ == "__main__":
    main()