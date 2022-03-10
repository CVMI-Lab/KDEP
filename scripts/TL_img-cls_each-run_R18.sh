#!/bin/sh

dataset=$1
ckpt_exp_name=$2
exp_name=$3
card=$4
lr=$5

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
  --lr ${lr} \
  --step-lr 50 \
  --batch-size 64 \
  --weight-decay 5e-4 \
  --adv-train 0 \
  --freeze-level -1 \
  --adv-eval 0 \
  --model-path outdir/${ckpt_exp_name}/checkpoint.pt.latest.attacker | tee outdir/${exp_name}/train-$now.log
