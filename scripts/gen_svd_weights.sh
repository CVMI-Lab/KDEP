#!/bin/sh
# sh scripts/gen_svd_weights.sh imgnet_128k ex_gen_svd 0

dataset=$1
exp_name=$2
card=$3

now=$(date +"%Y%m%d_%H%M%S")
pwd
mkdir -p outdir/${exp_name}

CUDA_VISIBLE_DEVICES=${card} python3 src/gen_svd_weights.py --arch resnet18 \
  --dataset ${dataset} \
  --data data/ \
  --out-dir outdir \
  --exp-name ${exp_name} \
  --epochs 150 \
  --lr 0.1 \
  --step-lr 50 \
  --batch-size 64 \
  --weight-decay 5e-4 \
  --adv-train 0 \
  --freeze-level -1 \
  --adv-eval 0 \
  --teacher_not_finetuned \
  --fc_classes 1000 \
  --teacher_path pretrained-models/MicrosoftVision-ResNet50.pth  \
  --teacher_arch resnet50_feat_pre_relu | tee outdir/${exp_name}/train-$now.log

#  --teacher_path pretrained-models/StandardSP-ImageNet1k-ResNet50.pth  \