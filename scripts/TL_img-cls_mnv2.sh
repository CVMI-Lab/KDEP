#!/bin/sh
# sh scripts/TL_img-cls_mnv2.sh exp_name

exp_name=$1

echo 'transform ckpt'
cd src/
python3 transform_ckpt_custom2attacker.py ${exp_name} 'mnv2'
cd ..

echo 'eval using card 0,1,2,3'
sh scripts/TL_img-cls_each-run_mnv2.sh caltech256 ${exp_name} ${exp_name}.caltech256.1 0 0.001 &
sh scripts/TL_img-cls_each-run_mnv2.sh caltech256 ${exp_name} ${exp_name}.caltech256.2 1 0.001 &
sh scripts/TL_img-cls_each-run_mnv2.sh dtd ${exp_name} ${exp_name}.dtd.1 2 0.001 &
sh scripts/TL_img-cls_each-run_mnv2.sh dtd ${exp_name} ${exp_name}.dtd.2 3 0.001 &
sh scripts/TL_img-cls_each-run_mnv2.sh cub ${exp_name} ${exp_name}.cub.1 2 0.001 &
sh scripts/TL_img-cls_each-run_mnv2.sh cub ${exp_name} ${exp_name}.cub.2 3 0.001 &
sh scripts/TL_img-cls_each-run_mnv2.sh cub ${exp_name} ${exp_name}.cub.lr0.01.1 2 0.01 &
sh scripts/TL_img-cls_each-run_mnv2.sh cub ${exp_name} ${exp_name}.cub.lr0.01.2 3 0.01 &
sh scripts/TL_img-cls_each-run_mnv2.sh cifar100 ${exp_name} ${exp_name}.cifar100.1 0,1 0.001 &
sh scripts/TL_img-cls_each-run_mnv2.sh cifar100 ${exp_name} ${exp_name}.cifar100.2 2,3 0.001

sleep 20m

# collect results
echo 'collect using card 0,1,2,3'
sh scripts/TL_img-cls_collect-results_mnv2.sh caltech256 ${exp_name}.1 ${exp_name}.caltech256.1 0 0.001 &
sh scripts/TL_img-cls_collect-results_mnv2.sh caltech256 ${exp_name}.2 ${exp_name}.caltech256.2 1 0.001 &
sh scripts/TL_img-cls_collect-results_mnv2.sh dtd ${exp_name}.3 ${exp_name}.dtd.1 2 0.001 &
sh scripts/TL_img-cls_collect-results_mnv2.sh dtd ${exp_name}.4 ${exp_name}.dtd.2 3 0.001 &
sh scripts/TL_img-cls_collect-results_mnv2.sh cub ${exp_name}.5 ${exp_name}.cub.1 2 0.001 &
sh scripts/TL_img-cls_collect-results_mnv2.sh cub ${exp_name}.6 ${exp_name}.cub.2 3 0.001 &
sh scripts/TL_img-cls_collect-results_mnv2.sh cub ${exp_name}.7 ${exp_name}.cub.lr0.01.1 2 0.01 &
sh scripts/TL_img-cls_collect-results_mnv2.sh cub ${exp_name}.8 ${exp_name}.cub.lr0.01.2 3 0.01 &
sh scripts/TL_img-cls_collect-results_mnv2.sh cifar100 ${exp_name}.9 ${exp_name}.cifar100.1 0 0.001 &
sh scripts/TL_img-cls_collect-results_mnv2.sh cifar100 ${exp_name}.10 ${exp_name}.cifar100.2 1 0.001

