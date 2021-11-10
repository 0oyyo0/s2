import os
import os.path as osp
from shutil import copyfile

def filterTxt(srcTxtPah, dstTxtPath, selected_class):
    selected_class_num = 0
    #  r:读取文件，若文件不存在则会报错
    with open(srcTxtPah, "r") as rf:
        for line in rf.readlines():
            line1 = line.strip().split(' ')
            if(selected_class == line1[0]):
                selected_class_num += 1
                #  a:写入文件,若文件不存在则会先创建再写入,但不会覆盖原文件,而是追加在文件末尾
                with open(dstTxtPath,"a") as af:
                    af.write(line)  # 自带文件关闭功能，不需要再写f.close()
    rf.close()
    return selected_class_num
    
#  DOTA数据的txt文件夹
txtFolder = r"/home/yy/project/s2anet/data/hjj_rssj_hrsc/train/labels"
#  DOTA数据的image文件夹
imgFolder = r"/home/yy/project/s2anet/data/hjj_rssj_hrsc/train/images"
#  要复制到的image文件夹
copy_imageFolder = r"/home/yy/project/s2anet/data/hjj_rssj_hrsc/1/images" # /home/yy/project/s2anet/data/dota_ship
#  要复制到的txt文件夹
copy_txtFolder = r"/home/yy/project/s2anet/data/hjj_rssj_hrsc/1/labels"
#  感兴趣类别
selected_class = "1"


if not osp.exists(copy_imageFolder):
        os.mkdir(copy_imageFolder)
if not osp.exists(copy_txtFolder):
        os.mkdir(copy_txtFolder)
txtNameList = os.listdir(txtFolder)
total=0
for i in range(len(txtNameList)):
    #  判断当前文件是否为txt文件
    if(os.path.splitext(txtNameList[i])[1] == ".txt"):
        txt_path = txtFolder + "/" + txtNameList[i]
        #  设置文件对象
        f = open(txt_path, "r")
        #  读取一行文件，包括换行符
        line = f.readline()
        line = line.strip().split(' ')
        while line:
            #  若该类是selected_class,则将对应图像复制粘贴,并停止循环
            if(selected_class == line[0]):
                #  获取txt的索引，不带扩展名的文件名
                txt_index = os.path.splitext(txtNameList[i])[0]
                #  获取对应图像文件的地址
                src = imgFolder + "/" + txt_index + ".tif"
                dst = copy_imageFolder + "/" + txt_index + ".tif"
                #  复制图像文件至指定位置
                copyfile(src, dst)
                #  筛选txt文件中的selected_class信息并写至指定位置
                selected_class_num = filterTxt(txt_path, copy_txtFolder + "/" + txt_index + ".txt", selected_class)
                # print(txt_index, ".png have", selected_class_num, selected_class)
                print('{}{} {} {}'.format(txt_index, ".tif have", selected_class_num, selected_class))
                total = total + 1
                break
            #  若第一行不是selected_class，继续向下读，直到读取完文件
            else:
                line = f.readline() 
f.close() #关闭文件
print(total) #total number