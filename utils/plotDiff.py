import os
import numpy as np
import cv2
from tqdm import tqdm
from shapely.geometry import Polygon

        
def extractTxt(labelFile):
    labels = []
    vertices = []
    with open(labelFile, 'r') as f:
        for line in f.readlines():
            lineSplit = line.split(' ')
            if len(lineSplit) < 9:
                continue
            labels.append(lineSplit[0])
            vertices.append([float(xy) for xy in lineSplit[1:9]])
    return np.array(vertices), labels


def plotGtBoxes(img, boxes, labels, colorMap):
    for box, label in zip(boxes, labels):
        pts = np.array(box, dtype=np.int32).reshape((1, 4, 2))
        cv2.polylines(img, pts, True, color=colorMap[label], thickness=3)
        cv2.putText(img, str(label), (int(box[0]), int(box[1])),
                    cv2.FONT_ITALIC, color=colorMap[label], fontScale=1, thickness=3)
    return img


def plotBoxes(img, boxes, labels, colorMap, badcase):
    for box, label in zip(boxes, labels):
        pts = np.array(box, dtype=np.int32).reshape((1, 4, 2))
        cv2.polylines(img, pts, True, color=colorMap[label], thickness=3)
        cv2.putText(img, str(label)+' '+badcase, (int(box[0]), int(box[1])),
                    cv2.FONT_ITALIC, color=colorMap[label], fontScale=1, thickness=3)
    return img


def loadDataset(imgPath, gtPath):
    imgFiles = [os.path.join(imgPath, img_file)
                for img_file in sorted(os.listdir(imgPath))]
    gtFiles = [os.path.join(gtPath, gt_file)
               for gt_file in sorted(os.listdir(gtPath))]
    data = []
    for img, gt in zip(imgFiles, gtFiles):
        boxes, labels = extractTxt(gt)
        data.append((img, boxes, labels))
    return data


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


def plot(colorMapD, colorMapG):
    data_gt = loadDataset("/home/yy/project/s2anet/data/hjj/train/images_png", '/home/yy/project/s2anet/data/hjj/train/labels')
    data_Pre = loadDataset("/home/yy/project/s2anet/data/hjj/train/images_png", '/home/yy/project/s2anet/data/hjj/train/labelsPre')
    for data_gt, data_Pre in tqdm(zip(data_gt, data_Pre), total=len(data_gt)):
        imgFile_gt, boxes_gt, labels_gt = data_gt
        imgFile_Pre, boxes_Pre, labels_Pre = data_Pre

        for i in range(0, len(boxes_gt)): 
            boxes1 = []
            labels1 = []
            boxes2 = []
            labels2 = []
            box_gt = []
            label_gt = []
            for j in range(0, len(boxes_Pre)):
                # print(boxes_gt[i], boxes_Pre[j])
                # print(labels_gt[i], labels_Pre[j])
                # if imgFile_gt[i] == imgFile_Pre[j] and iou(boxes_gt[i], boxes_Pre[j]) >= 0.7 and labels_gt[i] == labels_Pre[j]:
                #     continue
                if iou(boxes_gt[i], boxes_Pre[j]) >= 0.7 and labels_gt[i] == labels_Pre[j]:
                    continue
                elif iou(boxes_gt[i], boxes_Pre[j]) >= 0.7 and labels_gt[i] != labels_Pre[j]:
                    # print(imgFile_Pre)
                    # print('else')
                    # print('----------'*3)
                    box_gt.append(boxes_gt[i])

                    label_gt.append(labels_gt[i])

                    boxes1.append(boxes_Pre[j])
                    
                    labels1.append(labels_Pre[j])

                else: 
                    
                    boxes2.append(boxes_gt[i])
                    
                    labels2.append(labels_gt[i])

                imgName = os.path.basename(imgFile_Pre)
                img = cv2.imread(imgFile_Pre)
                if len(boxes1) != 0:
                    img = plotGtBoxes(img, box_gt, label_gt, colorMapG)
                    img = plotBoxes(img, boxes1, labels1, colorMapD, 'fp')
                if len(boxes2) != 0:
                    img = plotBoxes(img, boxes2, labels2, colorMapD, 'omit')
                cv2.imwrite(os.path.join(
                    '/home/yy/project/s2anet/tmp/test1/', imgName).split('.')[0]+'.png', img) 
                

if __name__ == '__main__':
    colorMapDiff = {
        '1': (255, 0, 0),
        '2': (0, 255, 0),
        '3': (0, 0, 255),
        '4': (128, 128, 0),
        '5': (255, 128, 128)
    }
    colorMapGt = {
        '1': (255, 255, 255),
        '2': (255, 255, 255),
        '3': (255, 255, 255),
        '4': (255, 255, 255),
        '5': (255, 255, 255)
    }
    
    plot(colorMapDiff, colorMapGt)
