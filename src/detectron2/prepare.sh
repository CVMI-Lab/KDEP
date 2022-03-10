git clone https://github.com/facebookresearch/detectron2.git


hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/proj/semseg_ws.tar
tar -xvf semseg_ws.tar
cd semseg
python3 -m pip install -e detectron2 -i https://pypi.tuna.tsinghua.edu.cn/simple

export DETECTRON2_DATASETS='/opt/tiger/filter_transfer/data/DETECTRON2_DATASETS/'
export DETECTRON2_DATASETS='/opt/tiger/semseg/dataset/DETECTRON2_DATASETS/'

hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/data/DETECTRON2_DATASETS_v1.tar data/

python3 train_net.py --num-gpus 8  --config-file ../configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml
python3 train_net.py --num-gpus 4  --config-file ../configs/PascalVOC-Detection/faster_rcnn_R_18_FPN.yaml --dist-url tcp://127.0.0.1:6777
python3 train_net.py --num-gpus 8  --config-file ../configs/PascalVOC-Detection/faster_rcnn_R_18_C4.yaml MODEL.WEIGHTS ../ckpts/resnet-18-l2-eps0.ckpt.clean.det.pkl
python3 train_net.py --num-gpus 8  --config-file ../configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml

python3 train_net.py --num-gpus 8  --config-file ../configs/COCO-Detection/faster_rcnn_R_18_C4_1x.yaml MODEL.WEIGHTS ../ckpts/resnet-18-l2-eps0.ckpt.clean.det.pkl


#python3 train_net.py --num-gpus 1  --config-file ../configs/Base-RCNN-mobilev2.yaml MODEL.WEIGHTS ../ckpts/ex99.1.t1.pt.det.pkl
python3 train_net.py --num-gpus 8  --config-file ../configs/COCO-Detection/faster_rcnn_mnv2-fpn_1x.yaml MODEL.WEIGHTS ../ckpts/ex99.1.t1.pt.det.pkl
python3 train_net.py --num-gpus 8  --config-file ../configs/PascalVOC-Detection/faster-rcnn_mobilev2.yaml MODEL.WEIGHTS ../ckpts/ex99.1.t1.pt.det.pkl


hdfs dfs -put exp/* hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/exp_filter/exp/

hdfs dfs -put *pkl hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/ckpt/det/
hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/ckpt/det/*

#hdfs dfs -mkdir -p hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/ckpt/det/