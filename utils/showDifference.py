import PIL.Image as Image
import numpy as np
import os
 

IMAGES_PATH_hrsc = r'/home/yy/project/s2anet/tmp/360_hrsc2016/'  # 图片集地址
IMAGES_PATH_dota = r'/home/yy/project/s2anet/tmp/360_dota/'  # 图片集地址

IMAGES_FORMAT = ['.tif', 'jpg']  # 图片格式
IMAGE_SIZE = 1024  # 每张小图片的大小
IMAGE_ROW = 2  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 2  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = r'/home/yy/project/s2anet/tmp/tmp/'  # 图片转换后的地址
 
# 获取图片集地址下的所有图片名称
image_names_hrsc = [name for name in os.listdir(IMAGES_PATH_hrsc) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
image_names_dota = [name for name in os.listdir(IMAGES_PATH_dota) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
print(image_names_hrsc)
print(image_names_dota)
img = Image.open(r"/home/yy/project/s2anet/tmp/360_hrsc2016/0_6656_128000.tif")
re_img = np.asarray(img)
print(re_img.dtype)
img_nrm = (img - np.min(img)) / (np.max(img) - np.min(img))
# array转回Image对象
 
# 显示方法一
# im = Image.fromarray(np.uint8(img))
 
#显示方法二，更合理
im = Image.fromarray(np.uint8(255*img_nrm))
im.show()
 
# # 简单的对于参数的设定和实际图片集的大小进行数量判断
# if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
#     raise ValueError("合成图片的参数和要求的数量不能匹配！")
 
# # 定义图像拼接函数
# def image_compose():
#     to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE)) #创建一个新图
#     # 循环遍历，把每张图片按顺序粘贴到对应位置上
#     for y in range(1, IMAGE_ROW + 1):
#         for x in range(1, IMAGE_COLUMN + 1):
#             from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
#                 (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
#             to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
#     return to_image.save(IMAGE_SAVE_PATH) # 保存新图


# if __name__ == '__main__':
#     image_compose() #调用函数