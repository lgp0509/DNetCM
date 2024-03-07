import os
import re
import  time
from tqdm import  *
from skimage import io
from skimage import transform
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


typelist = ["normal","mild","severe","nodular","sclerosis"]
for ty in typelist:
    foldpath = "C:/DenseNet-with-pytorch-main/data/val/"  # 目录 x:/xx/xx/
    savepath = "C:/DenseNet-with-pytorch-main/data/resize/val/"
    foldpath = foldpath  + ty +"/"
    savepath = savepath +ty + "/"
    mkdir(savepath)
    glofiles = os.listdir(foldpath)

    for img in tqdm(glofiles):
        img_path = foldpath + img
        img_ori = io.imread(img_path)
        img_trans = transform.resize(img_ori, (512, 512))
        io.imsave(savepath + img, img_trans)