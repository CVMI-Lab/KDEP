#!/bin/sh

# sh scripts/KDEP_MS-R50_mnv2.sh imgnet_128k exp_name 0,1,2,3


dataset=$1
exp_name=$2
card=$3

now=$(date +"%Y%m%d_%H%M%S")
pwd
mkdir -p outdir/${exp_name}


CUDA_VISIBLE_DEVICES=${card} python3 src/main.py --arch mobilenet_pre_relu \
 --dataset ${dataset} \
 --data data/ \
 --out-dir outdir \
 --exp-name ${exp_name} \
 --epochs 90 \
 --lr 0.1 \
 --step-lr 30 \
 --optimizer_custom sgd \
 --batch-size 512 \
 --weight-decay 5e-4 \
 --adv-train 0 \
 --freeze-level -1 \
 --adv-eval 0 \
 --loss_w 3.0 \
 --teacher_not_finetuned \
 --fc_classes 1000 \
 --teacher_path pretrained-models/MicrosoftVision-ResNet50.pth  \
 --teacher_arch resnet50_feat_svd | tee outdir/${exp_name}/train-$now.log


cp scripts/KDEP_MS-R50_mnv2.sh outdir/${exp_name}/
