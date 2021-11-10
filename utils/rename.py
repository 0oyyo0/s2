# --** coding="UTF-8" **--
# 
# author:SueMagic  time:2019-01-01
import os
import os.path as osp
import re
import sys
from shutil import copyfile


imgPath = r"/home/yy/project/s2anet/tmp/1019_trainset_infer1"
dstPath_img = r'/home/yy/project/s2anet/tmp/1019_trainset_infer'
# labelPath = r'/home/yy/project/s2anet/data/hjj_rssj_hrsc/1/labels'
# dstPath_label = r'/home/yy/project/s2anet/tmp/1_4/labels'
img_names = [osp.splitext(img_name.strip())[0] for img_name in os.listdir(imgPath)]
print(img_names)
if not osp.exists(dstPath_img):
        os.mkdir(dstPath_img)
# if not osp.exists(dstPath_label):
#         os.mkdir(dstPath_label)

for img_name in img_names:

    # img =  osp.join(imgPath, str(int(img_name)+2008)+'.tif')


    src_img = osp.join(imgPath, img_name+'.tif')
    # dst_img = osp.join(dstPath_img, str(int(img_name)+2008)+'.tif')
    dst_img = osp.join(dstPath_img, img_name+'.png')
    #  复制图像文件至指定位置
    copyfile(src_img, dst_img)
    print(src_img)
    print(dst_img)

#     src_labels = osp.join(labelPath, img_name+'.txt')
#     dst_labels = osp.join(dstPath_label, img_name+'_4'+'.txt')
#     #  复制图像文件至指定位置
#     copyfile(src_labels, dst_labels)
#     print(src_labels)
#     print(dst_labels)
