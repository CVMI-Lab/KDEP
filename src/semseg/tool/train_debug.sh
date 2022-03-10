#!/bin/sh
# usage:
# sh tool/train_debug.sh cityscapes config-psp18.yaml ex_seg_10.1
# sh tool/train_debug.sh voc2012 config-psp18.yaml ex_seg_10.2
# sh tool/train_debug.sh voc2012 bg.yaml ex_seg_bg
# sh tool/train_debug.sh ade20k config-psp18.yaml ex_seg_10.3

# sh tool/train_debug.sh cityscapes config-psp50.yaml ex_seg_7.1.1
# sh tool/train_debug.sh voc2012 config-psp50.yaml ex_seg_7.3.1
# sh tool/train_debug.sh ade20k config-psp50.yaml ex_seg_7.2.1

# sh tool/train_debug.sh cityscapes config-psp-KD.yaml ex_seg_8.1
# sh tool/train_debug.sh voc2012 config-psp-KD.yaml ex_seg_8.2
# sh tool/train_debug.sh ade20k config-psp-KD.yaml ex_seg_8.3

PYTHON=python3

dataset=$1
cfg=$2
exp_name=$3
ckpt_name=$4
train_gpu=$5
test_gpu=$6

echo ${exp_name}
echo ${ckpt_name}
echo ${train_gpu}
echo ${test_gpu}

ckpt_name=initmodel/${ckpt_name}
exp_dir=exp/${exp_name}
result_dir=${exp_dir}/result
model_dir=${exp_dir}/model
config=config/${dataset}/${cfg}
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}
mkdir -p ${exp_dir}
mkdir -p ${model_dir}
cp tool/train.sh tool/train.py tool/train_func.py tool/test.py model/pspnet.py model/deeplabv2.py ${config} ${exp_dir}


export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export PYTHONPATH=./
export KMP_INIT_AT_FORK=FALSE
$PYTHON -u ${exp_dir}/train.py --config=${exp_dir}/${cfg} --exp_name=${exp_name} --ckpt_name=${ckpt_name} train_gpu ${train_gpu} test_gpu ${test_gpu} 2>&1 | tee ${exp_dir}/train-$now.log

cd /opt/tiger/semseg/
hdfs dfs -put exp/${exp_name} hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/exp_filter/