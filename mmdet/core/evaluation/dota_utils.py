import os
import os.path as osp

from ..bbox import rbox2poly_single

# dotahbb原先的
# def result2dota_task1(results, dst_path, dataset):
#     CLASSES = dataset.CLASSES
#     img_names = dataset.img_names
#     assert len(results) == len(
#         img_names), 'length of results must equal with length of img_names'
#     if not osp.exists(dst_path):
#         os.mkdir(dst_path)
#     for classname in CLASSES:
#         f_out = open(osp.join(dst_path, 'Task1_'+classname+'.txt'), 'w')
#         print('Task1_'+classname+'.txt')
#         # per result represent one image
#         for img_id, result in enumerate(results):
#             for class_id, bboxes in enumerate(result):
#                 if CLASSES[class_id] != classname:
#                     continue
#                 if(bboxes.size != 0):
#                     for bbox in bboxes:
#                         score = bbox[5]
#                         bbox = rbox2poly_single(bbox[:5])
#                         temp_txt = '{} {} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
#                             osp.splitext(img_names[img_id])[0], score, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6], bbox[7])
#                         f_out.write(temp_txt)
#         f_out.close()
#     return True


# def result2dota_task2(results, dst_path, dataset):
#     CLASSES = dataset.CLASSES
#     img_names = dataset.img_names
#     if not osp.exists(dst_path):
#         os.mkdir(dst_path)
#     for classname in CLASSES:
#         f_out = open(osp.join(dst_path, 'Task2_'+classname+'.txt'), 'w')
#         print('Task2_'+classname+'.txt')
#         # per result represent one image
#         for img_id, result in enumerate(results):
#             filename = img_names[img_id]
#             filename = osp.basename(filename)
#             filename = osp.splitext(filename)[0]
#             for class_id, bboxes in enumerate(result):
#                 if CLASSES[class_id] != classname:
#                     continue
#                 if(bboxes.size != 0):
#                     for bbox in bboxes:
#                         score = bbox[4]
#                         temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
#                             filename, score, bbox[0], bbox[1], bbox[2], bbox[3])
#                         f_out.write(temp_txt)
#         f_out.close()
#     return True

# 修改为hjj
def result2dota_task1(results, dst_path, dataset):
    CLASSES = dataset.CLASSES
    img_names = dataset.img_names
    assert len(results) == len(
        img_names), 'length of results must equal with length of img_names'
    if not osp.exists(dst_path):
        os.mkdir(dst_path)
    for classname in CLASSES:
        f_out = open(osp.join(dst_path, 'Task1_'+classname+'.txt'), 'w')
        print('Task1_'+classname+'.txt')
        # per result represent one image
        for img_id, result in enumerate(results):
            for class_id, bboxes in enumerate(result):
                if CLASSES[class_id] != classname:
                    continue
                if(bboxes.size != 0):
                    for bbox in bboxes:
                        score = bbox[5]
                        if(score >= 0.1):
                            bbox = rbox2poly_single(bbox[:5])
                            temp_txt = '{}.tif {} {:.2f} {} {} {} {} {} {} {} {}\n'.format(
                                osp.splitext(img_names[img_id])[0], int(classname), score, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4]), int(bbox[5]), int(bbox[6]), int(bbox[7]))
                            f_out.write(temp_txt)
                        # if(classname == '1' and score >= 0.35):
                        #     bbox = rbox2poly_single(bbox[:5])
                        #     temp_txt = '{}.tif {} {:.2f} {} {} {} {} {} {} {} {}\n'.format(
                        #         osp.splitext(img_names[img_id])[0], int(classname), score, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4]), int(bbox[5]), int(bbox[6]), int(bbox[7]))
                        #     f_out.write(temp_txt)
                        # if(classname == '2' and score >= 0.40):
                        #     bbox = rbox2poly_single(bbox[:5])
                        #     temp_txt = '{}.tif {} {:.2f} {} {} {} {} {} {} {} {}\n'.format(
                        #         osp.splitext(img_names[img_id])[0], int(classname), score, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4]), int(bbox[5]), int(bbox[6]), int(bbox[7]))
                        #     f_out.write(temp_txt)
                        # if(classname == '3' and score >= 0.4):
                        #     bbox = rbox2poly_single(bbox[:5])
                        #     temp_txt = '{}.tif {} {:.2f} {} {} {} {} {} {} {} {}\n'.format(
                        #         osp.splitext(img_names[img_id])[0], int(classname), score, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4]), int(bbox[5]), int(bbox[6]), int(bbox[7]))        
                        #     f_out.write(temp_txt)
                        # if(classname == '4' and score >= 0.45):
                        #     bbox = rbox2poly_single(bbox[:5])
                        #     temp_txt = '{}.tif {} {:.2f} {} {} {} {} {} {} {} {}\n'.format(
                        #         osp.splitext(img_names[img_id])[0], int(classname), score, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4]), int(bbox[5]), int(bbox[6]), int(bbox[7]))
                        #     f_out.write(temp_txt)
                        # if(classname == '5' and score >= 0.43):
                        #     bbox = rbox2poly_single(bbox[:5])
                        #     temp_txt = '{}.tif {} {:.2f} {} {} {} {} {} {} {} {}\n'.format(
                        #         osp.splitext(img_names[img_id])[0], int(classname), score, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4]), int(bbox[5]), int(bbox[6]), int(bbox[7]))
                        #     f_out.write(temp_txt)
        f_out.close()
    return True


def result2dota_task2(results, dst_path, dataset):
    CLASSES = dataset.CLASSES
    img_names = dataset.img_names
    if not osp.exists(dst_path):
        os.mkdir(dst_path)
    for classname in CLASSES:
        f_out = open(osp.join(dst_path, 'Task2_'+classname+'.txt'), 'w')
        print('Task2_'+classname+'.txt')
        # per result represent one image
        for img_id, result in enumerate(results):
            filename = img_names[img_id]
            filename = osp.basename(filename)
            filename = osp.splitext(filename)[0]
            for class_id, bboxes in enumerate(result):
                if CLASSES[class_id] != classname:
                    continue
                if(bboxes.size != 0):
                    for bbox in bboxes:
                        score = bbox[4]
                        temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                            filename, score, bbox[0], bbox[1], bbox[2], bbox[3])  # classname，
                        f_out.write(temp_txt)
        f_out.close()
    return True
