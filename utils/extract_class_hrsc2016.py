import os
import os.path as osp
from shutil import copyfile


def filterTxt(srcTxtPah, dstTxtPath, selected_class):
    #  r:读取文件，若文件不存在则会报错
    with open(srcTxtPah, "r") as rf:
        for line in rf.readlines():
            if(selected_class in line):
                #  a:写入文件,若文件不存在则会先创建再写入,但不会覆盖原文件,而是追加在文件末尾
                with open(dstTxtPath,"a") as af:
                    af.write(line)  # 自带文件关闭功能，不需要再写f.close()
    rf.close()


def extract_class(imgFolder, txtFolder, copy_imageFolder, copy_txtFolder, selected_class):
    if not osp.exists(copy_imageFolder):
        os.mkdir(copy_imageFolder)
    if not osp.exists(copy_txtFolder):
        os.mkdir(copy_txtFolder)
    txtNameList = os.listdir(txtFolder)
    
                #  获取对应图像文件的地址
    for i in range(len(txtNameList)):
        if(os.path.splitext(txtNameList[i])[1] == '.txt'):
            txt_Path = txtFolder + '/' + txtNameList[i]
            print(txt_Path)
            f = open(txt_Path, 'r')
            line = f.readline()
            line = line.strip().split(' ')
            # print('reading')
            # while line:
            # print(line)
            if(len(line)<8):
                continue
            if(line[8] in selected_class):
                print(txtNameList[i])
                print(line)
            # else:
            #     continue
        # else:
        #     continue
                txt_index = os.path.splitext(txtNameList[i])[0]
                src = imgFolder + "/" + txt_index + ".bmp"
                dst = copy_imageFolder + "/" + txt_index + ".bmp"
                #  复制图像文件至指定位置
                copyfile(src, dst)
                filterTxt(txt_Path, copy_txtFolder + "/" + txt_index + ".txt", selected_class)
                # print('{}{} {} {}'.format(txt_index, ".png have", selected_class_num, selected_class))
                # total = total + 1
                # break
            #  若第一行不是selected_class，继续向下读，直到读取完文件
            # else:
            #     line = f.readline()

                


if __name__ == '__main__':
    imgFolder = r"/home/yy/project/s2anet/data/HRSC2016/FullDataSet/AllImages"  #  DOTA数据的image文件夹
    txtFolder = r"/home/yy/project/s2anet/data/HRSC2016/FullDataSet/labelTxt_all"  #  DOTA数据的txt文件夹
    copy_imageFolder = r"/home/yy/project/s2anet/tmp/test/images"  #  要复制到的image文件夹# /home/yy/project/s2anet/data/dota_ship
    copy_txtFolder = r"/home/yy/project/s2anet/tmp/test/labelTxt"  #  要复制到的txt文件夹
    selected_class = ['100000013', '100000005', '100000002', '100000006', '100000012', '100000031', '100000032', '100000033']  #  感兴趣类别
    # selected_class = ['100000002'] 
    for selected_class in selected_class:
        extract_class(imgFolder, txtFolder, copy_imageFolder, copy_txtFolder, selected_class)