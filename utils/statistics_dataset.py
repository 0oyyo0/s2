from shapely.geometry import Polygon
import numpy as np
import cv2
from PIL import Image
import math
import os
import xml.etree.cElementTree as ET
import torch
import torchvision.transforms as transforms
from torch.utils import data
import time
import logging
from utils import addImg, mulImg, drawMask, plotBoxes, calDistance, plotHist, plotBar
from config import config

logger = logging.getLogger()


def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Output:
        vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = calDistance(
        vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices


def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Output:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(calDistance(x1, y1, x2, y2), calDistance(x1, y1, x4, y4))
    r2 = min(calDistance(x2, y2, x1, y1), calDistance(x2, y2, x3, y3))
    r3 = min(calDistance(x3, y3, x2, y2), calDistance(x3, y3, x4, y4))
    r4 = min(calDistance(x4, y4, x1, y1), calDistance(x4, y4, x3, y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if calDistance(x1, y1, x2, y2) + calDistance(x3, y3, x4, y4) > \
       calDistance(x2, y2, x3, y3) + calDistance(x1, y1, x4, y4):
        offset = 0  # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1  # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:	
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def clockwiseVertices(vertices):
    '''clockwise the four coordinates of the polygon, 
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        clockwise vertices
    '''
    poly = vertices.reshape((4, 2))
    bottomIndex = np.argmax(poly[:, 1])
    bottomRightIndex = (bottomIndex - 1) % 4
    bottomLeftIndex = (bottomIndex + 1) % 4
    if poly[bottomRightIndex, 0] - poly[bottomLeftIndex, 0] < 0:
        poly = poly[::-1, :]
    return poly.reshape(-1)


def sortVertices(vertices):
    '''sort the four coordinates of the polygon, 
    default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        clockwise vertices
    '''
    poly = clockwiseVertices(vertices).reshape((4, 2))
    p_lowest = np.argmax(poly[:, 1])
    angle = 0.
    if np.count_nonzero(abs(poly[:, 1] - poly[p_lowest, 1]) < 1.) == 2:
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
    else:
        p_lowest_right = (p_lowest - 1) % 4
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right]
                            [1])/(poly[p_lowest][0] - poly[p_lowest_right][0] + 1e-5))
        if angle <= 0:
            logger.debug('angle: {:.8f}'.format(angle))
            logger.debug(poly)
        if angle/np.pi * 180 > 45 or angle <= 0:
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            angle = -(np.pi/2 - angle)
        else:
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
    orientedPoly = poly[[p0_index, p1_index, p2_index, p3_index]].reshape(-1)
    angle = angle / np.pi * 180.
    return orientedPoly, angle


def aroundVertices(vertices):
    '''get the around vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (4, 2)>
    Output:
        the around vertices as type int
    '''
    xMin, xMax, yMin, yMax = get_boundary(vertices.reshape(-1))
    xy = []
    for x, y in vertices:
        if abs(x-xMin) < 1e-5:
            x = np.ceil(x)
        elif abs(x-xMax) < 1e-5:
            x = np.floor(x)
        else:
            x = np.around(x)

        if abs(y-yMin) < 1e-5:
            y = np.ceil(y)
        elif abs(y-yMax) < 1e-5:
            y = np.floor(y)
        else:
            y = np.around(y)
        xy.append([x, y])
    return np.array(xy, dtype=np.int32)


def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = calDistance(x1, y1, x_min, y_min) + calDistance(x2, y2, x_max, y_min) + \
        calDistance(x3, y3, x_max, y_max) + calDistance(x4, y4, x_min, y_max)
    return err


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    angle_interval = 0.5
    angle_list = list(np.arange(-45, 45, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(
        list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
    '''check if the crop image crosses text regions
    Input:
        start_loc: left-top position
        length   : length of crop image
        vertices : vertices of text regions <numpy.ndarray, (n,8)>
    Output:
        True if crop image crosses text region
    '''
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w + length, start_h,
                  start_w + length, start_h + length, start_w, start_h + length]).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / (p2.area+1e-5) <= 0.99:
            return True
    return False


def checkAndValidatePolys(vertices, labels, length, minThreshold=0.6):
    newVertices = []
    newLabels = []
    imgPoly = np.array([0, 0, length, 0, length, length,
                        0, length]).reshape((4, 2))
    imgPoly = Polygon(imgPoly)
    for v, label in zip(vertices, labels):
        poly = Polygon(v.reshape((4, 2)))
        if poly.is_valid:
            inter = imgPoly.intersection(poly).area
            if inter > 1:
                if inter / poly.area < minThreshold:
                    label = -1
                newLabels.append(label)
                newVertices.append(v)
    return np.array(newVertices), np.array(newLabels)


def crop_img(img, vertices, labels, length, maxTry=10):
    '''crop img patches to obtain batch and augment
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        labels      : -1->ignore, <numpy.ndarray, (n,)>
        length      : length of cropped image region
    Output:
        region      : cropped image region
        new_vertices: new vertices in cropped region
    '''
    h, w = img.height, img.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert(ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    remain_h = img.height - length
    remain_w = img.width - length
    if vertices.size > 0:
        new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h

        # find random position
        flag = True
        cnt = 0
        while flag and cnt < maxTry:
            cnt += 1
            start_w = int(np.random.rand() * remain_w)
            start_h = int(np.random.rand() * remain_h)
            flag = is_cross_text([start_w, start_h], length,
                                 new_vertices[labels >= 0, :])
        new_vertices[:, [0, 2, 4, 6]] -= start_w
        new_vertices[:, [1, 3, 5, 7]] -= start_h
    else:
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    return region, new_vertices


def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Output:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
        np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def adjustHeight(img, vertices, ratio=0.2):
    '''adjust height of image to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:, [1, 3, 5, 7]] *= new_h / old_h
    return img, new_vertices


def rotate_img(img, vertices, angle_range=10):
    '''rotate image [-10, 10] degree to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        angle_range : rotate range
    Output:
        img         : rotated PIL Image
        new_vertices: rotated vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i, :] = rotate_vertices(
            vertice, -angle / 180 * math.pi, np.array([[center_x], [center_y]]))
    return img, new_vertices


def adjustScale(img, vertices, ratio=2.):
    '''scale image to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : scale changes in [1/scaleRange, scaleRange]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    assert ratio >= 1
    scale = 1. + (ratio - 1.) * np.random.rand()
    if np.random.rand() > 0.5:
        scale = 1. / scale

    oldW = img.width
    oldH = img.height
    w = int(np.around(oldW * scale))
    h = int(np.around(oldH * scale))
    img = img.resize((w, h), Image.BILINEAR)
    newVertices = vertices.copy()
    if vertices.size > 0:
        newVertices[:, [1, 3, 5, 7]] *= h / oldH
        newVertices[:, [0, 2, 4, 6]] *= w / oldW
    return img, newVertices


def horizontalFlip(img, vertices, prob=0.5):
    '''filp image left to right
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        prob        : flip prob
    Output:
        img         : fliped PIL Image
        new_vertices: filped vertices
    '''
    if np.random.rand() > (1. - prob):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        new_vertices = np.zeros(vertices.shape)
        for i, vertice in enumerate(vertices):
            vertice = vertice.reshape((4, 2))
            vertice[:, 0] = img.width - 1 - vertice[:, 0]
            new_vertices[i, :] = vertice[::-1, :].reshape(-1)
        return img, new_vertices
    else:
        return img, vertices


def verticalFlip(img, vertices, prob=0.5):
    '''filp image top to bottom
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        prob        : vertical prob
    Output:
        img         : fliped PIL Image
        new_vertices: fliped vertices
    '''
    if np.random.rand() > (1. - prob):
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        new_vertices = np.zeros(vertices.shape)
        for i, vertice in enumerate(vertices):
            vertice = vertice.reshape((4, 2))
            vertice[:, 1] = img.height - 1 - vertice[:, 1]
            new_vertices[i, :] = vertice[::-1, :].reshape(-1)
        return img, new_vertices
    else:
        return img, vertices


def getScoreAndGeo(img, vertices, labels, scale, length, numCat):
    '''generate score gt and geometry gt
    Input:
        img     : PIL Image
        vertices: vertices of text regions <numpy.ndarray, (n,8)>
        labels  : -1->ignore, <numpy.ndarray, (n,)>
        scale   : feature map / image
        length  : image length
    Output:
        score gt, geo gt, ignored
        geo gt is sort as:
                    cat0 | cat1| cat2 | ...
            top   | 
            bottom|
            left  |
            right |
            theta | 
    '''
    startTime = time.time()
    scoreMap = np.zeros(
        (int(img.height * scale), int(img.width * scale), numCat), np.float32)
    geoMap = np.zeros(
        (int(img.height * scale), int(img.width * scale), 5*numCat), np.float32)
    ignoredMap = np.zeros(
        (int(img.height * scale), int(img.width * scale), 1), np.float32)

    index = sorted(range(len(vertices)), key=lambda k: Polygon(
        vertices[k].reshape((4, 2))).area, reverse=True)
    sortedVertices = vertices[index]
    sortedLabels = labels[index]

    ignoredPolys = []
    scorePolys = {i: [] for i in range(numCat)}
    for vertice, label in zip(sortedVertices, sortedLabels):
        vertice, _ = sortVertices(vertice)

        if label == -1:
            ignoredPolys.append(
                aroundVertices(scale * vertice.reshape((4, 2))))
            continue

        poly = aroundVertices(scale * shrink_poly(vertice).reshape((4, 2)))
        scorePolys[label].append(poly)

        polyMask = np.zeros(scoreMap.shape[:-1], np.float32)
        cv2.fillPoly(polyMask, [poly], 1)

        theta = find_min_rect_angle(vertice)
        rotate_mat = get_rotate_mat(theta)

        rotated_vertices = rotate_vertices(vertice, theta)
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
        rotatedX, rotatedY = rotate_all_pixels(
            rotate_mat, vertice[0]*scale, vertice[1]*scale, length*scale)
        rotatedX /= scale
        rotatedY /= scale

        d1 = rotatedY - y_min
        d1[d1 < 0] = 0
        d2 = y_max - rotatedY
        d2[d2 < 0] = 0
        d3 = rotatedX - x_min
        d3[d3 < 0] = 0
        d4 = x_max - rotatedX
        d4[d4 < 0] = 0
        geoMap[:, :, 0*numCat + label] = d1 * polyMask + \
            (1 - polyMask) * geoMap[:, :, 0*numCat + label]
        geoMap[:, :, 1*numCat + label] = d2 * polyMask + \
            (1 - polyMask) * geoMap[:, :, 1*numCat + label]
        geoMap[:, :, 2*numCat + label] = d3 * polyMask + \
            (1 - polyMask) * geoMap[:, :, 2*numCat + label]
        geoMap[:, :, 3*numCat + label] = d4 * polyMask + \
            (1 - polyMask) * geoMap[:, :, 3*numCat + label]
        geoMap[:, :, 4*numCat + label] = theta * polyMask + \
            (1 - polyMask) * geoMap[:, :, 4*numCat + label]

    for i in range(numCat):
        if len(scorePolys[i]) > 0:
            polyMask = np.zeros(scoreMap.shape[:-1], np.float32)
            cv2.fillPoly(polyMask, scorePolys[i], 1)
            scoreMap[:, :, i] = polyMask
    cv2.fillPoly(ignoredMap, ignoredPolys, 1)
    logger.debug('done time is {:.8f}'.format(time.time()-startTime))
    return torch.Tensor(scoreMap).permute(2, 0, 1), torch.Tensor(geoMap).permute(2, 0, 1), torch.Tensor(ignoredMap).permute(2, 0, 1)


def extractTxt(labelFile, catToLabel, ignoreHard=False):
    '''extract vertices info from txt lines
    Input:
        labelFile :the path of label
        catToLabel: a dict, map category to index
    Output:
        vertices  : vertices of text regions <numpy.ndarray, (n,8)>
        labels    : -1->ignore, <numpy.ndarray, (n,)>
    '''
    labels = []
    vertices = []
    with open(labelFile, 'r') as f:
        for line in f.readlines():
            lineSplit = line.split(' ')
            if len(lineSplit) < 9:
                continue
            if ignoreHard and lineSplit[9] == 1:
                label = -1
            else:
                label = catToLabel[lineSplit[8]]
            labels.append(label)
            vertices.append([float(xy) for xy in lineSplit[:8]])
    return np.array(vertices), np.array(labels)


def extractXml(labelFile, catToLabel, ignoreHard=False):
    boxes = []
    labels = []
    for element in ET.parse(labelFile).getroot():
        if element.tag == 'object':
            name = element.find('name').text
            difficult = int(element.find('difficult').text)
            box = [float(line.text) for line in element.find('bndbox')]
            if ignoreHard and difficult == 1:
                label = -1
            else:
                label = catToLabel[name]
            labels.append(label)
            boxes.append(box)
    return np.array(boxes), np.array(labels)


class CustomDataset(data.Dataset):
    def __init__(self, imgPath, gtPath, scale=0.25, length=512, plot=False):
        super(CustomDataset, self).__init__()
        self.scale = scale
        self.length = length
        self.plot = plot
        self.categoriesInv = {k: i for i, k in enumerate(config.categories)}
        self.numCat = len(config.categories)
        self.data = self.load(imgPath, gtPath)

    def load(self, imgPath, gtPath):
        imgFiles = [os.path.join(imgPath, img_file)
                    for img_file in sorted(os.listdir(imgPath))]
        gtFiles = [os.path.join(gtPath, gt_file)
                   for gt_file in sorted(os.listdir(gtPath))]
        data = []
        for img, gt in zip(imgFiles, gtFiles):
            vertices, labels = extractXml(gt, self.categoriesInv)
            data.append((img, vertices, labels))
        return data

    def __len__(self):
        return len(self.data)

    def fetch(self):
        return self[np.random.randint(0, len(self.data)-1)]

    def __getitem__(self, index):
        imgFile, verticesOri, labelsOri = self.data[index]
        vertices = verticesOri.copy()
        labels = labelsOri.copy()
        logger.debug('num instance is {:.8f}'.format(vertices.shape[0]))

        img = Image.open(imgFile)
        img, vertices = adjustHeight(img, vertices)
        img, vertices = horizontalFlip(img, vertices, prob=0.5)
        img, vertices = verticalFlip(img, vertices, prob=0.5)
        img, vertices = rotate_img(img, vertices, angle_range=30.)
        img, vertices = adjustScale(img, vertices, ratio=1.3)
        img, vertices = crop_img(img, vertices, labels, self.length)
        vertices, labels = checkAndValidatePolys(vertices, labels, self.length)

        if self.plot:
            for vertice in vertices:
                plotBoxes(img, [vertice])
                plotBoxes(img, [shrink_poly(vertice)])
            img.save('./statistics/debug.jpg')

        transform = transforms.Compose([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=config.mean, std=config.std)])

        scoreMap, geoMap, ignoredMap = getScoreAndGeo(
            img, vertices, labels, self.scale, self.length, self.numCat)
        return transform(img), scoreMap, geoMap, ignoredMap


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    dataset = CustomDataset(config.train_img_path,
                            config.train_gt_path, length=config.length, plot=True)

    vertices = []
    catStat = {k: 0 for k in config.categories}
    catStat['ignore'] = 0
    for _, v, labels in dataset.data:
        vertices += v.tolist()
        for label in labels:
            if label == -1:
                catName = 'ignore'
            else:
                catName = config.categories[label]
            catStat[catName] += 1
    plotHist(vertices)
    plotBar(catStat)

    for _ in range(100):
        img, scoreMap, geoMap, ignoredMap = dataset[np.random.randint(
            0, len(dataset)-1)]

        img = img.permute(1, 2, 0).numpy()
        img = addImg(mulImg(img, config.std), config.mean)*256
        cv2.imwrite('./statistics/img.jpg', img)

        scoreMap = scoreMap.permute(1, 2, 0).numpy()*256
        for i in range(scoreMap.shape[-1]):
            color = (np.random.randint(0, 256), np.random.randint(
                0, 256), np.random.randint(0, 256))
            _tmp = cv2.resize(
                scoreMap[:, :, i], img.shape[:-1], interpolation=cv2.INTER_NEAREST)
            img = drawMask(img, _tmp > 128, color)
        cv2.imwrite('./statistics/score.jpg', img)

        ignoredMap = ignoredMap.permute(1, 2, 0).numpy()*256
        cv2.imwrite('./statistics/ignored.jpg', ignoredMap)

        geoMap = geoMap.permute(1, 2, 0).numpy()
        d1, d2, d3, d4, angle = np.split(geoMap, 5, 2)
        cv2.imwrite('./statistics/d1.jpg', np.sum(np.abs(d1), axis=2))
        cv2.imwrite('./statistics/d2.jpg', np.sum(np.abs(d2), axis=2))
        cv2.imwrite('./statistics/d3.jpg', np.sum(np.abs(d3), axis=2))
        cv2.imwrite('./statistics/d4.jpg', np.sum(np.abs(d4), axis=2))
        cv2.imwrite('./statistics/angle.jpg',
                    np.sum(np.abs(angle), axis=2) /np.pi * 256)
