# Knowledge Distillation as Efficient Pretraining: Faster Convergence, Higher Data-efficiency, and Better Transferability

This repository contains the code and models necessary to replicate the results of our paper:

```bibtex
@inproceedings{he2022knowledge,
  title={Knowledge Distillation as Efficient Pre-training: Faster Convergence, Higher Data-efficiency, and Better Transferability
},
  author={He, Ruifei and Sun, Shuyang, and Yang, Jihan, and Bai, Song and Qi, Xiaojuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Abstract

Large-scale pre-training has been proven to be crucial for various computer vision tasks.
However, with the increase of pre-training data amount, model architecture amount, and the private/inaccessible data, it is not very efficient or possible to pre-train all the model architectures on large-scale datasets. In this work, we investigate an alternative strategy for pre-training, namely Knowledge Distillation as Efficient Pre-training (**KDEP**), aiming to efficiently transfer the learned feature representation from existing pre-trained models to new student models for future downstream tasks. We observe that existing Knowledge Distillation (KD) methods are unsuitable towards pre-training since they normally distill the logits that are going to be discarded when transferred to downstream tasks. To resolve this problem, we propose a feature-based KD method with non-parametric feature dimension aligning. Notably, our method performs comparably with supervised pre-training counterparts in 3 downstream tasks and 9 downstream datasets requiring **10x** less data and **5x** less pre-training time.

![pics-cropped-1](pics-cropped-1.png)

## Getting started
1.  Clone our repo: `git clone https://github.com/CVMI-Lab/KDEP.git`

2.  Install dependencies:
    ```sh
    conda create -n KDEP python=3.7
    conda activate KDEP
    pip install -r requirements.txt
    ```

## Data preparation

* ImageNet-1K ([Download](https://www.image-net.org/)) 
* Caltech256 ([Download](http://www.vision.caltech.edu/Image_Datasets/Caltech256/))
* Cifar100 **(Automatically downloaded when you run the code)**
* DTD ([Download]( https://www.robots.ox.ac.uk/~vgg/data/dtd/))
* CUB-200 ([Download](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html))
* Cityscapes ([Download](https://www.cityscapes-dataset.com/))
* VOC (segmentation and detection, [Download](http://host.robots.ox.ac.uk/pascal/VOC/))
* ADE20K ([Download](http://groups.csail.mit.edu/vision/datasets/ADE20K/))
* COCO ([Download](https://cocodataset.org/))

For image classification datasets (except for Caltech256), the folder structure should follow ImageNet:

```
data root
├─ train/
  ├── n01440764
  │   ├── n01440764_10026.JPEG
  │   ├── n01440764_10027.JPEG
  │   ├── ......
  ├── ......
├─ val/
  ├── n01440764
  │   ├── ILSVRC2012_val_00000293.JPEG
  │   ├── ILSVRC2012_val_00002138.JPEG
  │   ├── ......
  ├── ......
```

For semantic segmentation datasets, please refer to [PyTorch Semantic Segmentation](https://github.com/hszhao/semseg).

For object detection datasets, please refer to [Detectron2](https://github.com/facebookresearch/Detectron2).

## Pre-training with KDEP

1. Download teacher models ([Download](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/ruifeihe_connect_hku_hk/Es4xT6QzDi9JtkFmmqXIWF4B-MAxG7z6oIDqtrZ88m5MpA?e=Jh19d1)), and put them under `pretrained-models/` .

2. You can use a provided python file `scripts/make-imgnet-subset.py` to create the 10% of ImageNet-1K data.

3. Update the path of the dataset for KDEP (10% or 100% of ImageNet-1K) in `src/utils/constants.py`.

4. Prepare the SVD weights for teacher models. You can download the weights we provide ([Download](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/ruifeihe_connect_hku_hk/Es4xT6QzDi9JtkFmmqXIWF4B-MAxG7z6oIDqtrZ88m5MpA?e=Jh19d1)) or generate using our provided script `scripts/gen_svd_weights.sh` .

   ```sh
   sh scripts/gen_svd_weights.sh imgnet_128k ex_gen_svd 0
   ```

5. Scripts of pre-training with KDEP are in `scripts/`. For example, you can use teacher-student pair of Microsoft ResNet50 -> ResNet18 with `scripts/KDEP_MS-R50_R18.sh` by:

   ```sh
   sh scripts/KDEP_MS-R50_R18.sh imgnet_128k exp_name 90 30 5e-4 0,1,2,3
   ### imgnet_128k or imgnet_full to select 10% or 100% ImageNet-1K data
   ### 90 is #epoch, 30 is step-lr
   ### 5e-4 is weight decay
   ### 0,1,2,3 is GPU id
   ```

   You can run KDEP with different data amount and training schedules by changing the data name (imgnet_128k or imgnet_full), #epoch and step-lr, and weight decay. 

   Note that we do not generate the svd weights for 100% ImageNet-1K data, but directly use the svd weights generated from 10% data.

## Transfer learning experiments

### Image classification

1. We use four image classification tasks: CIFAR100, DTD, Caltech256, CUB-200. 

2. Scripts (`scripts/TL_img-cls_R18.sh` and `scripts/TL_img-cls_mnv2.sh` ) are provided for running all  four tasks twice for a distilled student (R18/mnv2). 

   ```sh
   sh scripts/TL_img-cls_R18.sh exp_name
   # note the exp_name here should be identical to that of the distilled student
   ```

### Semantic segmentation

1. We use three semantic segmentation tasks: Cityscapes, VOC2012, ADE20K.

2. Transform the checkpoint into segmentation code format by `src/transform_ckpt_custom2seg.py` 

   ```sh
   cd src
   python3 transform_ckpt_custom2seg.py exp_name
   # note the exp_name here should be identical to that of the distilled student
   ```

   Move the transformed checkpoint to `semseg/initmodel/`.

3. Scripts (`semseg/tool/TL_seg_R18.sh` and `semseg/tool/TL_seg_mnv2.sh` ) are provided for running all three tasks twice for a distilled student (R18/mnv2). 

   ```sh
   cd semseg
   sh tool/TL_seg_R18.sh ckpt_name
   # note the ckpt_name should be what you put into the semseg/initmodel/ in step1.
   ```

### Object detection

1. We use two object detection tasks: COCO and VOC.

2. Transform the checkpoint into Detectron2 format by `src/transform_ckpt_custom2det.py` 

   ```sh
   cd src
   python3 transform_ckpt_custom2det.py exp_name R18
   # note the exp_name here should be identical to that of the distilled student
   # R18 could be changed to mnv2
   ```

   Move the transformed checkpoint to `detectron2/ckpts/` .

3. Install Detectron2, and export dataset path

   ```sh
   python3 -m pip install -e detectron2
   export DETECTRON2_DATASETS='path/to/datasets'
   ```

4. Scripts (`detectron2/tool/TL_det_R18.sh` and `detectron2/tool/TL_det_mnv2.sh` ) are provided for running all two tasks twice for a distilled student (R18/mnv2). 

   ```sh
   cd detectron2/tool
   sh TL_det_R18.sh ckpt_name
   # note the ckpt_name should be what you put into the semseg/initmodel/ in step1.
   ```

## Distilled models of KDEP
We provide some distilled models of KDEP here. 
1. ([Download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/ruifeihe_connect_hku_hk/ES1ZvPYyRRlAoMdtwwYh_d0B7Sfl1ghBMs09mLEHVY5HqA?e=P7dpdJ)) ResNet18, KDEP(SVD+PTS) from MS-R50 teacher on 100% ImageNet-1K data for 90 epochs.
1. ([Download](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/ruifeihe_connect_hku_hk/EeteR1gfIJZAqIypbG0LgJwBJ5sRT-GAvWU8M2WOkUlsHA?e=PiN74K)) MobileNet-V2, KDEP(SVD+PTS) from MS-R50 teacher on 100% ImageNet-1K data for 90 epochs.


## Acknowledgement

Our code is mainly based on  [robust-models-transfer](https://github.com/microsoft/robust-models-transfer), we also thank the open source code from [PyTorch Semantic Segmentation](https://github.com/hszhao/semseg) and [Detectron2](https://github.com/facebookresearch/Detectron2).

