#!/bin/sh
# sh tool/TL_seg_R18.sh ckpt_name
ckpt_name=$1
save_name='eval_seg_'${ckpt_name}
sh tool/train_debug.sh cityscapes config-psp18.yaml ${save_name}.cs1 ${ckpt_name} 0,1,2,3 [0] &
sh tool/train_debug.sh cityscapes config-psp18.yaml ${save_name}.cs2 ${ckpt_name} 4,5,6,7 [2] &
sh tool/train_debug.sh voc2012 config-psp18.yaml ${save_name}.voc1 ${ckpt_name} 0,1,2,3 [1] &
sh tool/train_debug.sh voc2012 config-psp18.yaml ${save_name}.voc2 ${ckpt_name} 4,5,6,7 [3] &
sh tool/train_debug.sh ade20k config-psp18.yaml ${save_name}.ade1 ${ckpt_name} 0,1,2,3 [4] &
sh tool/train_debug.sh ade20k config-psp18.yaml ${save_name}.ade2 ${ckpt_name} 4,5,6,7 [6] &

