from torchvision import transforms
from .Randaug import RandAugment

prefix = "data"

# ImageNet 10%
# IMGNET_128k_PATH = prefix + "/imagenet/ILSVRC2012_img_train_1000cls_128shot_128k/"
IMGNET_128k_PATH = prefix +  "/imagenet/ILSVRC2012_img_train_128k/"

# ImageNet 100%
IMGNET_FULL_PATH = prefix + "/imagenet/ILSVRC2012_img_train/"

# DTD dataset
DTD_PATH = prefix + "/dtd/"

# CUB-200-2011 birds
CUB_PATH = prefix + "/CUB_200_2011"

# Caltech datasets
CALTECH256_PATH = prefix + ""

value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]

# Data Augmentation defaults
TRAIN_TRANSFORMS = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            # transforms.RandomResizedCrop(224, scale=(0.08,1.0), ratio=(0.75,1.333333)),
            # transforms.RandomResizedCrop(224, scale=(0.08,1.0), ratio=(0.5,2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])

TEST_TRANSFORMS = transforms.Compose([
        # transforms.Resize(32),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),
    ])
