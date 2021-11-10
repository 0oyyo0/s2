import os
import os.path as osp
from PIL import Image
from tqdm import tqdm

original_path = '/home/yy/project/s2anet/data/hjj_rssj_hrsc/train/images'
saved_path = u'/home/yy/project/s2anet/tmp/1219'

counts = 0
if not osp.exists(saved_path):
        os.mkdir(saved_path)
for fileName in tqdm(os.listdir(original_path)):
    if fileName.endswith('tif'):
        
        im = Image.open(os.path.join(original_path, fileName))

        new_fileName = fileName[:-3] + 'png'
        im.save(os.path.join(saved_path, new_fileName))
        counts += 1

print('%d done' %counts)