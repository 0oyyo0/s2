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
                    with open(os.path.join(txtNow, txtName + '.txt'), "a") as af:
                            af.write('{} {} {} {} {} {} {} {} {}'.format(line1[1], line1[3], line1[4], line1[5], line1[6], line1[7], line1[8], line1[9], line1[10]))
                    print(os.path.join(txtNow, txtName + '.txt'))
                    a+=1
                    line = f.readline()
    print(a)


def parse_gt(filename):
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 8):
                    continue

                object_struct['category'] = splitlines[0]
                object_struct['bbox'] = [float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7]),
                                         float(splitlines[8])]
                objects.append(object_struct)
            else:
                break

    return objects


def iou(g, p):
    g=np.asarray(g)
    p=np.asarray(p)
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union
   

def box(txt, imagename):
    filename = txt.format(imagename)
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                if (len(splitlines) < 8):
                    continue

                # object_struct['category'] = splitlines[0]
                BOX = [float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7]),
                                         float(splitlines[8])]
                # objects.append(object_struct)
            else:
                break

    return BOX


def omission(txtPre, txtNow, imagename):
    BB = box(txtPre, imagename)
    bb = box(txtNow, imagename)
    print(BB)
    print(bb)
    a = iou(BB, bb)
    print(a)



if __name__ == '__main__':
    txtSrc = r"/home/yy/project/s2anet/data/hjj/train/1024_train/1024_train.txt"
    txtPre = r'/home/yy/project/s2anet/data/hjj/train/labels/{:s}.txt'
    txtNow = r"/home/yy/project/s2anet/tmp/tmp/{:s}.txt"
    filterTxt(txtSrc, '/home/yy/project/s2anet/data/hjj/train/labelsPre')
    # for a, b in zip(iou, a):
    # ob1 = parse_gt('/home/yy/project/s2anet/data/hjj/train/labels/1.txt')
    # print(ob1)
    # imagenames = '1'
    # omission(txtPre, txtNow, imagenames)