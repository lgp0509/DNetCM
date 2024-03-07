import os
import re
import  time
from tqdm import  *
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import openslide
from openslide.deepzoom import DeepZoomGenerator
import warnings
warnings.filterwarnings('ignore')


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False

foldpath = "Z:/DN_Slide/"#目录 x:/xx/xx/
sildelist = os.listdir(foldpath)

for img in sildelist:
    if img[-5:] == ".mrxs":
        img_path = foldpath + img
        slide = openslide.open_slide(img_path)
        data_gen = DeepZoomGenerator(slide, tile_size=512, overlap=0, limit_bounds=True)

        down_level = 3  # 起始减几级
        for level_offset in range(1):  # 往上几级
            level = data_gen.level_count - down_level + level_offset
            mkdir("Z:\\result\\" + img + "\\" + str(level) + "\\")
            result_path = "Z:\\result\\" + img + "\\" + str(level) + "\\"
            [i_range, j_range] = data_gen.level_tiles[level - 1]

            for i in tqdm(range(i_range)):
                for j in range(j_range):
                    tile_img = data_gen.get_tile(level - 1, (i, j))
                    if tile_img.getcolors() is None:
                        tile_array = np.array(tile_img)
                        io.imsave(result_path + "/"  + img[:-5]  + "_"+ str(level) + "_" + str(i) + "_" + str(j) + ".png", tile_array)  # 保存图像