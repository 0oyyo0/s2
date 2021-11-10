import cv2
from PIL import Image, ImageDraw
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


def mulImg(img, scale):
    img[:, :, 0] *= scale[0]
    img[:, :, 1] *= scale[1]
    img[:, :, 2] *= scale[2]
    return img


def addImg(img, scale):
    img[:, :, 0] += scale[0]
    img[:, :, 1] += scale[1]
    img[:, :, 2] += scale[2]
    return img


def drawBox(im, box, color=(0, 255, 0)):
    x0, y0, x1, y1 = box
    im = cv2.rectangle(im, (int(x0), int(y0)),
                       (int(x1), int(y1)), color, 2)
    return im


def drawMask(im, mask, color=(0, 255, 0)):
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    im[:, :, 0][mask] = 0.5*im[:, :, 0][mask] + 0.5*color[0]
    im[:, :, 1][mask] = 0.5*im[:, :, 1][mask] + 0.5*color[1]
    im[:, :, 2][mask] = 0.5*im[:, :, 2][mask] + 0.5*color[2]
    return im


def drawPred(im, boxes, masks=None):
    numInstances = len(boxes)
    for i in range(numInstances):
        color = (np.random.randint(0, 256), np.random.randint(
            0, 256), np.random.randint(0, 256))
        im = drawBox(im, boxes[i], color=color)
        if masks:
            im = drawMask(im, masks[i], color=color)
    return im


def plotBoxes(img, boxes, outline=(0, 255, 0)):
    '''plot boxes on image
    '''
    if boxes is None:
        return img

    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon([box[0], box[1], box[2], box[3], box[4],
                      box[5], box[6], box[7]], outline=outline)
        draw.text((box[0], box[1]), '0')
        draw.text((box[2], box[3]), '1')
        draw.text((box[4], box[5]), '2')
        draw.text((box[6], box[7]), '3')
    return img


def calDistance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def plotHist(vertices, **kwargs):
    Dist = []
    Area = []
    for v in np.array(vertices):
        x1, y1, x2, y2, x3, y3, x4, y4 = v
        d1 = calDistance(x1, y1, x2, y2)
        d2 = calDistance(x2, y2, x3, y3)
        d3 = calDistance(x3, y3, x4, y4)
        d4 = calDistance(x4, y4, x1, y1)
        Dist += [d1, d2, d3, d4]
        Area.append(Polygon(v.reshape((4, 2))).area)

    plt.cla()
    plt.hist(Dist, bins=512, range=(0, 128), **kwargs)
    plt.savefig('./statistics/length-hist.jpg')

    plt.cla()
    plt.hist(Area, bins=1000, range=(0, 10000))
    plt.savefig('./statistics/area-hist.jpg')
    return


def plotBar(data):
    cell = []
    Y = []
    for k, v in data.items():
        cell.append(k)
        Y.append(v)
    X = np.arange(len(data))
    plt.cla()
    plt.bar(X, Y, width=0.5)
    plt.xticks(X, cell, fontsize=8, rotation=15, horizontalalignment='right')
    plt.savefig('./statistics/bar.jpg')
    return
