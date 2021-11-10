import os
from shutil import copyfile


def find_class(txtFolder, class_name):
    txtNameList = os.listdir(txtFolder)
    a = 0
    for i in range(len(txtNameList)):
        if(os.path.splitext(txtNameList[i])[1] == '.txt'):
            txt_Path = txtFolder+'/'+txtNameList[i]
            f = open(txt_Path, 'r')
            line = f.readline()
            
            line = line.strip().split(' ')
            # print('reading')
            # while line:
            # print(line)
            if(len(line)<8):
                continue
            elif(line[0] in class_name):
                print(txtNameList[i])
                print(line)
                a = a+1

            # else:
            #     continue
        # else:
        #     continue
    return a
                


if __name__ == '__main__':
    txtFolder = r"/home/yy/project/s2anet/data/hjj_rssj_hrsc_1plus/train/labels"
    # class_name = ['100000013', '100000005', '100000002', '100000006', '100000012', '100000031', '100000032', '100000033', ]
    class_name = ['5']
    count = find_class(txtFolder, class_name)
    print(count)
