import os
import numpy as np
import cv2
from tqdm import tqdm
from shapely.geometry import Polygon

def extractTxt(labelFile):  # 解析txt文件中的记录
    labels = []
    vertices = []
    with open(labelFile, 'r') as f:
        for line in f.readlines():
            lineSplit = line.split(' ')
            if len(lineSplit) < 9:
                continue
            labels.append(lineSplit[8])
            vertices.append([float(xy) for xy in lineSplit[0:8]])
    return np.array(vertices), labels


def plotGtBoxes(img, boxes, labels, colorMap):  # 根据label在图片上画框
    for box, label in zip(boxes, labels):
        pts = np.array(box, dtype=np.int32).reshape((1, 4, 2))
        cv2.polylines(img, pts, True, color=colorMap[label], thickness=3)
        cv2.putText(img, str(label), (int(box[0]), int(box[1])),
                    cv2.FONT_ITALIC, color=colorMap[label], fontScale=1, thickness=3)
    return img


def loadDataset(imgPath, gtPath):  # 加载图片以及对应的label
    imgFiles = [os.path.join(imgPath, img_file)
                for img_file in sorted(os.listdir(imgPath))]
    gtFiles = [os.path.join(gtPath, gt_file)
               for gt_file in sorted(os.listdir(gtPath))]
    data = []
    for img, gt in zip(imgFiles, gtFiles):
        boxes, labels = extractTxt(gt)
        data.append((img, boxes, labels))
    return data


def plotGt(colorMap):
    data = loadDataset("/home/yy/project/s2anet/data/dota/train/images", '/home/yy/project/s2anet/data/dota/train/labelTxt')
    for (imgFile, boxes, labels) in tqdm(data, total=len(data)):
        imgName = os.path.basename(imgFile)
        img = cv2.imread(imgFile)
        img = plotGtBoxes(img, boxes, labels, colorMap)
        cv2.imwrite(os.path.join(
            '/home/yy/project/s2anet/tmp/dota/', imgName).split('.')[0]+'.png', img)


if __name__ == '__main__':
    colorMapGt = {
        'plane': (54, 67, 244),
        'baseball-diamond': (99, 30, 233),
        'bridge': (176, 39, 156),
        'ground-track-field': (183, 58, 103),
        'small-vehicle': (181, 81, 63),
        'large-vehicle': (243, 150, 33),
        'ship': (212, 188, 0),
        'tennis-court': (80, 175, 76),
        'basketball-court': (74, 195, 139),
        'storage-tank': (57, 220, 205),
        'soccer-ball-field': (59, 235, 255),
        'roundabout': (0, 152, 255),
        'harbor': (0, 0, 255),
        'swimming-pool': (255, 255, 255),
        'helicopter': (34, 87, 255),
        'container-crane': (72, 85, 121) 
        
    }
    
    plotGt(colorMapGt)
