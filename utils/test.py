import os
import os.path
import os.path as osp
import xml.etree.ElementTree as ET
import glob
import xmltodict


def xml_to_txt(xmlpath, txtpath, img_path):
    if not osp.exists(txtpath):
        os.mkdir(txtpath)
    img_names = [osp.splitext(img_name.strip())[0] for img_name in os.listdir(img_path)]
    for img_name in img_names:
        label = osp.join(xmlpath, img_name+'.xml')
        label_txt = osp.join(txtpath, img_name+'.txt')
        print(label_txt)
        f_label = open(label)
        data_dict = xmltodict.parse(f_label.read())
        # print(data_dict)
        # print('----------')
        data_dict = data_dict['annotation']
        # print(data_dict)
        data_dict = data_dict['object']
        # print('----------')
        # print(data_dict)
        f_label.close()
        # os.chdir(xmlpath)

        # annotations = os.listdir('.')
        # annotations = glob.glob(str(annotations)+'*.xml')
        file_xml = os.path.join(xmlpath, img_name+'.xml')
        
        # print(annotations)

        # for i,file in enumerate(annotations):

        in_file = open(file_xml)
        file_txt = os.path.join(txtpath, img_name+'.txt')
        f_w = open(file_txt, 'w')
        tree=ET.parse(in_file)
        root = tree.getroot()
        # filename = root.find('filename').text
        for obj in root.iter('object'):
                # current = list()
                # name = obj.find('name').text

                # class_num = class_names.index(name)

                xmlbox = obj.find('bndbox')
                x1 = xmlbox.find('x1').text
                y1 = xmlbox.find('y1').text
                x2 = xmlbox.find('x2').text
                y2 = xmlbox.find('y2').text
                x3 = xmlbox.find('x3').text
                y3 = xmlbox.find('y3').text
                x4 = xmlbox.find('x4').text
                y4 = xmlbox.find('y4').text

                difficult = obj.find('difficult').text

                f_w.write(x1+' '+y1+' '+x2+' '+y2+' '+x3+' '+y3+' '+x4+' '+y4+' '+'ship'+' '+difficult+'\n')


if __name__ == "__main__":
    class_names = ['2']
    img_path='/home/yy/project/s2anet/utils/test/imgs'
    xmlpath='/home/yy/project/s2anet/utils/test/labels'
    txtpath='/home/yy/project/s2anet/utils/test/labeltxt'
    xml_to_txt(xmlpath, txtpath, img_path)

