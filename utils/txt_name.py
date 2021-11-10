import os
import os.path
import os.path as osp
import xml.etree.ElementTree as ET
import glob
import xmltodict


def name_to_txt(txtpath, img_path):
    if not osp.exists(txtpath):
        os.mkdir(txtpath)
    img_names = [osp.splitext(img_name.strip())[0] for img_name in os.listdir(img_path)]
    file_txt = os.path.join(txtpath, 'train.txt')
    f_w = open(file_txt, 'w')
    for img_name in img_names:
        
        f_w.write(img_name+'\n')


if __name__ == "__main__":
    class_names = ['2']
    img_path='/home/yy/project/s2anet/data/hjj/train/images'
    # xmlpath='/home/yy/project/s2anet/data/360_pretrain/Train/labels'
    txtpath='/home/yy/project/s2anet/data/hjj/train/labels'
    name_to_txt(txtpath, img_path)