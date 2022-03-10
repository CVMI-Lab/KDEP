#!/bin/sh
# sh TL_det_R18.sh ckpt_name
ckpt_name=$1
save_name='eval_det_'${ckpt_name}

mkdir exp

python3 train_net.py --num-gpus 8  --config-file ../configs/COCO-Detection/faster_rcnn_R_18_C4_1x.yaml MODEL.WEIGHTS ../ckpts/${ckpt_name} OUTPUT_DIR ./exp/${save_name}.coco1
python3 train_net.py --num-gpus 8  --config-file ../configs/COCO-Detection/faster_rcnn_R_18_C4_1x.yaml MODEL.WEIGHTS ../ckpts/${ckpt_name} OUTPUT_DIR ./exp/${save_name}.coco2
python3 train_net.py --num-gpus 8  --config-file ../configs/PascalVOC-Detection/faster_rcnn_R_18_C4.yaml MODEL.WEIGHTS ../ckpts/${ckpt_name} OUTPUT_DIR ./exp/${save_name}.voc1
python3 train_net.py --num-gpus 8  --config-file ../configs/PascalVOC-Detection/faster_rcnn_R_18_C4.yaml MODEL.WEIGHTS ../ckpts/${ckpt_name} OUTPUT_DIR ./exp/${save_name}.voc2
