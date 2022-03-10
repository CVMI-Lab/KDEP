#!/bin/sh
cd /opt/tiger/semseg/
mkdir dataset
mkdir initmodel
hdfs dfs -get  hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/data/cityscapes.zip dataset/
hdfs dfs -get  hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/data/ade20k.zip dataset/
hdfs dfs -get  hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/data/voc12.tar dataset/

#hdfs dfs -get  hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/data/initmodel/resnet50_v2.pth initmodel/
#hdfs dfs -get  hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/data/initmodel/ex53.1.t1_latest.seg initmodel/
#hdfs dfs -get  hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/data/initmodel/resnet-18-l2-eps0.ckpt.custom.seg initmodel/

hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/data/initmodel/* initmodel/
# hdfs dfs -put initmodel/* hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/data/initmodel/
# hdfs dfs -put pretrained-models/seg/* hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/data/initmodel/
# hdfs dfs -put *.seg hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/data/initmodel/

#hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/user/sunshuyang.kevin/checkpoints/vit/prog_trans_qpos_sinpos_dpsk3_p100_warm20_bs1024/model_best.pth.tar initmodel/
cd dataset
unzip cityscapes.zip
unzip ade20k.zip
tar -xvf voc12.tar
mv voc12 voc2012
cd ..

sh tool/train_debug.sh voc2012 bg.yaml ex_seg_bg

# ---- proj saving
hdfs dfs -put semseg_new.tar hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/proj/

# --- get exp results
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/exp_filter/eval_ex*pt.seg*/

hdfs dfs -rm -r hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/exp_filter/eval_ex101.4.e3.pt.seg*/model
hdfs dfs -rm -r hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/exp_filter/eval_ex101.4.e3.pt.seg*/result
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/exp_filter/eval_ex101.4.e3.pt.seg*/

