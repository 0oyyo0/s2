import xml.etree.ElementTree as ET
import os
import numpy as np
import matplotlib.pyplot as plt
import polyiou
from functools import partial
import pdb
import os.path as osp


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
                object_struct['name'] = splitlines[0]
                object_struct['difficult'] = 0
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


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
            # cachedir,
             ovthresh=0.7,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    #if not os.path.isdir(cachedir):
     #   os.mkdir(cachedir)
    #cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # print('imagenames: ', imagenames)
    #if not os.path.isfile(cachefile):
        # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        #print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    
    image_ids = [x[0] for x in splitlines]

    image_ids = [i.strip().split('.')[0] for i in image_ids]
    # print(image_ids)

    confidence = np.array([float(x[2]) for x in splitlines])

    # print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[3:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)

    sorted_scores = np.sort(-confidence)

    # print('check sorted_scores: ', sorted_scores)
    # print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    # print('check imge_ids: ', image_ids)
    # print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    # print('check fp:', fp)
    print('number of fp:', sum(fp==1))
    # print('check tp', tp)
    print('number of tp:', sum(tp==1))
    print('npos num:', npos)
    print('漏检率:', (npos-sum(tp==1))/npos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def main(score, iou):
    detpath = '/home/yy/project/s2anet/data/hjj/train/1029/result_raw{}/'.format(score) + r'Task1_{:s}.txt'  # 选择score0.1的目录       
    annopath = r'/home/yy/project/s2anet/data/hjj/train/labels/{:s}.txt'  # 原始标注文件
    imagesetfile = r'/home/yy/project/s2anet/data/hjj/train/train.txt'   

    classnames = ['1', '2', '3', '4', '5']

    map = 0
    classaps = []
    plt.figure(figsize=(8,4))
    print(f'iou={iou}, score={score}')
    plt.title(f'iou={iou}, score={score}')
    plt.xlabel('recall')
    plt.ylabel('precision')
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
            annopath,
            imagesetfile,
            classname,
            ovthresh=iou,
            use_07_metric=True)
        map = map + ap
        # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        classaps.append(ap)

        plt.plot(rec, prec, label = f"{classname}")
        plt.legend(loc='upper left')


    if not osp.exists(f'/home/yy/project/s2anet/tmp/fig1108/respective'):
        os.mkdir(f'/home/yy/project/s2anet/tmp/fig1108/respective')
    plt.savefig(f'/home/yy/project/s2anet/tmp/fig1108/respective/iou={iou}andscore={score}.png')
    plt.close()


    map = map/len(classnames)
    print(f'when score={score}:')
    print('map:', map)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)
    return map



if __name__ == '__main__':
    # score = 0.1
    # iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # a = []
    # for iou in iou:
    #     map = main(score, iou)
    #     a.append(map)
    # # print(a)
    # iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # plt.figure(figsize=(8,4))
    # plt.title("map-iou")
    # plt.xlabel('iou')
    # plt.ylabel('map')
    # plt.plot(score, a)
    # for a, b in zip(iou, a):
    #     plt.text(a, b, '{:.5f}'.format(b))
    # plt.savefig('/home/yy/project/s2anet/tmp/fig/map-iou.png')


    score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    iou = 0.7
    a = []
    for score in score:
        map = main(score, iou)
        a.append(map)
    print(a)
    score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plt.figure(figsize=(8,4))
    plt.title("map-score")
    plt.xlabel('score')
    plt.ylabel('map')
    plt.plot(score, a)
    for a, b in zip(score, a):
        plt.text(a, b, '{:.3f}'.format(b))
    plt.savefig('/home/yy/project/s2anet/tmp/fig1108/map-score.png')


    # plt.figure()

    # ax = plt.gca()

    # y = np.random.randn(9)

    # col_labels = ['预测值1', '预测值2', '预测值3', '预测值4', '预测值5']

    # row_labels = ['真实值1', '真实值2', '真实值3', '真实值4', '真实值5']

    # table_vals = [[11,12,13],[21,22,23],[28,29,30]]


    # my_table = plt.table(cellText=table_vals, colWidths=[0.1]*3, rowLabels=row_labels,  loc='best')

    # plt.plot(y)
