import os
from tqdm.autonotebook import tqdm
from tqdm import trange
from math import ceil
import numpy
from PIL import Image, ImageDraw
import geojson
import shapely
from shapely.geometry import shape
from shapely.strtree import STRtree
from shapely.geometry import Point
from shapely.geometry import Polygon
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
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

for type in ["train","val"]:
    path = f'C:/yolov9/datasets/glo2/labels/{type}/'  # "E:/DN_Slide/"切片地址
    dir = os.listdir(path)
    for patch in dir:
        # 定义原始文本文件路径
        file_path = path + patch

        # 打开文件并读取内容
        with open(file_path, 'r') as file:
            content = file.read()

        # 对内容进行修改（这里只是将每行前添加了一个字符串）
        modified_content = "0 " + content[2:]

        # 重新写入修改后的内容到同名文件
        with open(file_path, 'w') as file:
            file.writelines(modified_content)

