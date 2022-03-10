#!/bin/sh

dataset=$1
exp_name=$2
ckpt_exp_name=$3
card=$4


now=$(date +"%Y%m%d_%H%M%S")
pwd
mkdir -p outdir/${exp_name}


# ------- res18
CUDA_VISIBLE_DEVICES=${card} python3 src/main.py --arch resnet18 \
  --dataset ${dataset} \
  --data data/ \
  --out-dir outdir \
  --exp-name ${exp_name} \
  --optimizer_custom sgd \
  --epochs 150 \
  --lr 1e-10 \
  --step-lr 50 \
  --batch-size 64 \
  --weight-decay 5e-4 \
  --adv-train 0 \
  --weight \
  --weight_path outdir/${ckpt_exp_name}/checkpoint.pt.best \
  --freeze-level -1 \
  --eval-only 1 \
  --adv-eval 0 | tee outdir/${exp_name}/train-$now.collect.log

