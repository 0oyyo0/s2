import os
import os.path as osp
import numpy as np
from shapely.geometry import Polygon

def filterTxt(txtSrc, txtNow):
    a = 0
    if not osp.exists(txtNow):
        os.mkdir(txtNow)
    with open(txtSrc, "r") as f:
            line = f.readline()
            while line:
                    line1 = line.split(' ')
                    txtName = line1[0][:-4]
                    with open(os.path.join(txtNow, '1103_1' + '.txt'), "a") as af:
                        print('{}.png {} {} {} {} {} {} {} {} {} {}'.format(txtName, line1[5], line1[6], line1[7], line1[8], line1[9], line1[10], line1[11], line1[12], line1[13], line1[14]))
                        af.write('{}.png {} {} {} {} {} {} {} {} {} {}'.format(txtName, line1[5], line1[6], line1[7], line1[8], line1[9], line1[10], line1[11], line1[12], line1[13], line1[14]))
                    print(os.path.join(txtNow, txtName + '.txt'))
                    a+=1
                    line = f.readline()
    print(a)



filterTxt('/home/yy/project/s2anet/data/hjj/train/1103/1103.txt', '/home/yy/project/s2anet/data/hjj/train/1103')
print('done')