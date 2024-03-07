import os
from tqdm.autonotebook import tqdm
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

path = "E:/DN_Slide/"
WSI_dir = os.listdir(path)
slidelist = []
for silde_name in WSI_dir:
    if silde_name[-5:] == ".mrxs":
        slidelist.append(silde_name)

for img in slidelist:
    json_fname = path + img + ".json"  # input geojson file
    if json_fname.endswith(".gz"):
        with gzip.GzipFile(json_fname, 'r') as f:
            allobjects = geojson.loads(f.read(), encoding='ascii')
    else:
        with open(json_fname) as f:
            allobjects = geojson.load(f)
    print("done loading")

    allshapes = [shape(obj["nucleusGeometry"] if "nucleusGeometry" in obj.keys() else obj["geometry"]) for obj in
                 allobjects]
    print("done converting")

    img_path = path + img
    slide = openslide.open_slide(img_path)

    data_gen = DeepZoomGenerator(slide, tile_size=512, overlap=0, limit_bounds=True) #读对应切片大小
    # print(data_gen.level_count)
    # print(data_gen.tile_count)
    # print(data_gen.level_tiles)
    # print(data_gen.level_dimensions)

    down_level = 3
    data_level = (data_gen.level_count - 1) - down_level
    slide_level_downsamples = data_gen.level_dimensions[data_level]
    [m1, n1] = data_gen.level_dimensions[data_level]

    down_level_shape = []
    for i in range(len(allshapes)):
        x, y = shapely.affinity.scale(allshapes[i], xfact=1 / (2 ** down_level), yfact=1 / (2 ** down_level),
                                      origin=(0, 0)).exterior.xy
        x = numpy.array(x).tolist()
        y = numpy.array(y).tolist()
        polypoint = []
        for n in range(len(x)):
            polypoint.append((x[n], y[n]))
            formatpolypoint = list(set(polypoint))
            formatpolypoint.sort(key=polypoint.index)
        down_level_shape.append(formatpolypoint)

    maskimg = Image.new('L', (m1, n1), 0)
    for i in range(len(down_level_shape)):
        ImageDraw.Draw(maskimg).polygon(down_level_shape[i], outline=255, fill=255)
    mask = numpy.array(maskimg)
    io.imsave(img_path + ".png", mask)

    png_path = img_path + ".png"
    maskslide = openslide.open_slide(png_path)
    data_gen = DeepZoomGenerator(maskslide, tile_size=512, overlap=0, limit_bounds=False)

    level = data_gen.level_count
    mkdir("E:\\result\\" + img + ".png" + "\\" + str(level) + "\\")
    result_path = "E:\\result\\" + img + ".png" + "\\" + str(level) + "\\"
    [i_range, j_range] = data_gen.level_tiles[level - 1]

    for i in tqdm(range(i_range)):
        for j in range(j_range):
            tile_img = data_gen.get_tile(level - 1, (i, j))
            tile_array = np.array(tile_img)
            io.imsave(result_path + "/" + img[:-5] + str(level) + "_" + str(i) + "_" + str(j) + ".png", tile_array)  # 保存图像