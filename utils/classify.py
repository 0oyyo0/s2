import os
import os.path as osp
from shutil import copyfile


def filterTxt(srcTxtPah, dstTxtPath, selected_class, score):
    selected_class_num = 0
    with open(srcTxtPah, "r") as rf:
        for line in rf.readlines():
            line1 = line.strip().split(' ')
            if(selected_class == line1[1] and float(line1[2]) >= score):
                selected_class_num += 1
                with open(dstTxtPath,"a") as af:
                    af.write(line) 
    rf.close()
    return selected_class_num


def classify(txt_path, copy_txtFolder, selected_class):
    f = open(txt_path, "r")

    line = f.readline()
    line = line.strip().split(' ')
    while line:
        if(selected_class == line[1]):

            selected_class_num = filterTxt(txt_path, copy_txtFolder + "/" + f'Task1_{selected_class}' + ".txt", selected_class, score)
            print(f'{selected_class}: ', selected_class_num)
            break

        else:
            line = f.readline() 
            
    f.close()

if __name__ == '__main__':
    selected_class = ['1', '2', '3', '4', '5']
    for score in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f'when score >= {score}')

        txt_path = r"/home/yy/project/s2anet/data/hjj/train/1109/1109.txt"  # 原始txt文件
        copy_txtFolder = f"/home/yy/project/s2anet/data/hjj/train/1109/result_raw{score}"  # 根据得分保存到不同文件夹，然后用evalution_hjj.py评估单个score下的map

        if not osp.exists(copy_txtFolder):
            os.mkdir(copy_txtFolder)

        for selected in selected_class:
            classify(txt_path, copy_txtFolder, selected)
        
