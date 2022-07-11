#!/bin/sh

# sh scripts/KDEP_MS-R50_R18.sh imgnet_128k exp_name 90 30 5e-4 0,1,2,3

dataset=$1
exp_name=$2
epochs=$3
step-lr=$4
weight_decay=$5
card=$6

now=$(date +"%Y%m%d_%H%M%S")
pwd
mkdir -p outdir/${exp_name}


##### ------- （current）res50->res18, res50: 2048->512, pre relu pca, r18: pre relu
CUDA_VISIBLE_DEVICES=${card} python3 src/main.py --arch resnet18_feat_pre_relu \
 --dataset ${dataset} \
 --data data/ \
 --out-dir outdir \
 --exp-name ${exp_name} \
 --epochs ${epochs} \
 --lr 0.3 \
 --step-lr ${step-lr} \
 --optimizer_custom sgd \
 --batch-size 512 \
 --weight-decay ${weight_decay} \
 --adv-train 0 \
 --freeze-level -1 \
 --adv-eval 0 \
 --loss_w 3.0 \
 --teacher_not_finetuned \
 --fc_classes 1000 \
 --teacher_path pretrained-models/MicrosoftVision-ResNet50.pth  \
 --teacher_arch resnet50_feat_svd | tee outdir/${exp_name}/train-$now.log


cp scripts/KDEP_MS-R50_R18.sh outdir/${exp_name}/
