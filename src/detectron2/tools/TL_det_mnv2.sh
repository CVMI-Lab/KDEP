#!/bin/sh
# sh TL_det_mnv2.sh ckpt_name
ckpt_name=$1
save_name='eval_det_'${ckpt_name}

mkdir exp

python3 train_net.py --num-gpus 8  --config-file ../configs/COCO-Detection/faster_rcnn_mnv2-fpn_1x.yaml MODEL.WEIGHTS ../ckpts/${ckpt_name} OUTPUT_DIR ./exp/${save_name}.coco1
python3 train_net.py --num-gpus 8  --config-file ../configs/COCO-Detection/faster_rcnn_mnv2-fpn_1x.yaml MODEL.WEIGHTS ../ckpts/${ckpt_name} OUTPUT_DIR ./exp/${save_name}.coco2
python3 train_net.py --num-gpus 8  --config-file ../configs/PascalVOC-Detection/faster-rcnn_mobilev2.yaml MODEL.WEIGHTS ../ckpts/${ckpt_name} OUTPUT_DIR ./exp/${save_name}.voc1
python3 train_net.py --num-gpus 8  --config-file ../configs/PascalVOC-Detection/faster-rcnn_mobilev2.yaml MODEL.WEIGHTS ../ckpts/${ckpt_name} OUTPUT_DIR ./exp/${save_name}.voc2


