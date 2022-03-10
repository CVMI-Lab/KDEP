# This file should be located in the train/ folder in ImageNet-1k dataset which has subfolders like "n15075141"

import glob
import os
import random

category_folders = sorted(glob.glob('n*'))
print(category_folders)

m=128  # shot number
cls=1000  # class number
total = m*cls/1000
m_shot_path_name = '_'+str(cls)+'cls_' +str(m)+'shot_'+str(int(total))+'k'
os.system('mkdir ../%s' % m_shot_path_name)

random.shuffle(category_folders)
category_folders = category_folders[:cls]

for category in category_folders:

    category_files = glob.glob('%s/*' % category)
    random.shuffle(category_files)
    copy_files = category_files[:m]
    os.system('mkdir ../{}/{}'.format(m_shot_path_name, category))
    # import ipdb
    # ipdb.set_trace(context=20)
    for i in range(m):
        os.system('cp {} ../{}/{}/'.format(copy_files[i], m_shot_path_name, category))
        # print('cp {} ../{}/{}/'.format(copy_files[i], m_shot_path_name, category))

os.system('mkdir ../../{}'.format("ILSVRC2012_img_train"+m_shot_path_name))
os.system('mkdir ../../{}/train'.format("ILSVRC2012_img_train"+m_shot_path_name))
os.system('mv ../{}/* ../../{}/train/'.format(m_shot_path_name, "ILSVRC2012_img_train"+m_shot_path_name))
os.system('cp -r ../val ../../{}/'.format("ILSVRC2012_img_train"+m_shot_path_name))