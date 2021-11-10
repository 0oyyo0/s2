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
        file_xml = os.path.join(xmlpath, img_name+'.xml')
        in_file = open(file_xml)
        file_txt = os.path.join(txtpath, img_name+'.txt')
        f_w = open(file_txt, 'w')
        tree=ET.parse(in_file)
        root = tree.getroot()
        for obj in root.iter('object'):
            name = obj.find('name').text
            xmlbox = obj.find('bndbox')
            x1 = float(xmlbox.find('x1').text)
            y1 = float(xmlbox.find('y1').text)
            x2 = float(xmlbox.find('x2').text)
            y2 = float(xmlbox.find('y2').text)
            x3 = float(xmlbox.find('x3').text)
            y3 = float(xmlbox.find('y3').text)
            x4 = float(xmlbox.find('x4').text)
            y4 = float(xmlbox.find('y4').text)
            
            # x1 = xmlbox.find('x1').text
            # y1 = xmlbox.find('y1').text
            # x2 = xmlbox.find('x2').text
            # y2 = xmlbox.find('y2').text
            # x3 = xmlbox.find('x3').text
            # y3 = xmlbox.find('y3').text
            # x4 = xmlbox.find('x4').text
            # y4 = xmlbox.find('y4').text

            difficult = obj.find('difficult').text
            # print(type(x1))
            # print(type(difficult))
            
            str_line = '{} {} {} {} {} {} {} {} {} {}\n'.format(x1, y1, x2, y2, x3, y3, x4, y4, name, difficult)
            f_w.write(str_line)  # x1+' '+y1+' '+x2+' '+y2+' '+x3+' '+y3+' '+x4+' '+y4+' '+'ship'+' '+difficult+'\n'


if __name__ == "__main__":
    img_path='/home/yy/project/s2anet/data/360_pretrain/Train/images'
    xmlpath='/home/yy/project/s2anet/data/360_pretrain/Train/labels'
    txtpath='/home/yy/project/s2anet/data/360_pretrain/Train/labelTxt'
    xml_to_txt(xmlpath, txtpath, img_path)

