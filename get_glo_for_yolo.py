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
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import os

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

def  slide_process(args):
    img, path = args

    json_fname = path + img + ".json"  # input geojson file
    with open(json_fname) as f:
        allobjects = geojson.load(f)

    # print("done loading")

    allshapes = [shape(obj["nucleusGeometry"] if "nucleusGeometry" in obj.keys() else obj["geometry"]) for obj in
                 allobjects]
    # print("done converting")

    img_path = path + img
    slide = openslide.open_slide(img_path)

    #data_gen = DeepZoomGenerator(slide, tile_size=512, overlap=0, limit_bounds=True)  # 读对应切片大小
    # print(slide.level_count)
    # print(slide.dimensions)
    # print(data_gen.level_count)
    # print(data_gen.level_dimensions)
    # print(data_gen.tile_count)
    # print(data_gen.level_tiles)
    # print(data_gen.level_dimensions)
    down_level = 0
    data_level = down_level  # level = 0 是原始图像，故与deepzoom的level不同

    for i in range(len(allshapes)):

        glo_pathology_type = allobjects[i].get('properties')
        if 'classification' in glo_pathology_type:
            glo_pathology_type = glo_pathology_type.get('classification').get('name')
        else:
            glo_pathology_type = 'notype'
        x, y = shapely.affinity.scale(allshapes[i], xfact=1 / (2 ** down_level), yfact=1 / (2 ** down_level),
                                      origin=(0, 0)).exterior.xy
        x = numpy.array(x).tolist()
        y = numpy.array(y).tolist()
        x_min = int(min(x))
        x_max = int(max(x))
        y_min = int(min(y))
        y_max = int(max(y))
        x_size = int((x_max - x_min) * 1.05)
        y_size = int((y_max - y_min) * 1.05)
        # polypoint = []
        # for n in range(len(x)):
        #     polypoint.append((x[n], y[n]))
        #     formatpolypoint = list(set(polypoint))
        #     formatpolypoint.sort(key=polypoint.index)
        #     down_level_shape.append(formatpolypoint)
        # down_level_shape = []

        prop = slide.properties
        bounds_x = int(prop.get('openslide.bounds-x'))
        bounds_y = int(prop.get('openslide.bounds-y'))

        ratio_expan = 0.5  # 向一个方向扩大的倍数
        x_start = int((bounds_x + x_min) - (x_size * ratio_expan))
        y_start = int((bounds_y + y_min) - (y_size * ratio_expan))

        x_size_expan = int(x_size * (1 + 2 * ratio_expan))
        y_size_expan = int(y_size * (1 + 2 * ratio_expan))

        glo_patch_rgba = slide.read_region((x_start, y_start), 0, (x_size_expan, y_size_expan))
        glo_patch = np.array(glo_patch_rgba.convert("RGB"))
        result_path = "C:\\result_glo_yolo\\"
        mkdir(result_path)
        io.imsave(result_path + "\\" + glo_pathology_type + "_" + img[:-5] + "_" + str(data_level) + "_" + str(
            i) + "_" + ".jpg", glo_patch)

        x_min_yolo = bounds_x + x_min - x_start
        x_max_yolo = bounds_x + x_max - x_start
        y_min_yolo = bounds_y + y_min - y_start
        y_max_yolo = bounds_y + y_max - y_start

        x_center = ((x_min_yolo + x_max_yolo) / 2 - 1) / x_size_expan
        y_center = ((y_min_yolo + y_max_yolo) / 2 - 1) / y_size_expan
        w_yolo = (x_max_yolo - x_min_yolo) / x_size_expan
        h_yolo = (y_max_yolo - y_min_yolo) / y_size_expan

        if glo_pathology_type == "Normal glo":
            type_yolo = 0
        elif glo_pathology_type == "Mes-p mild":
            type_yolo = 1
        elif glo_pathology_type == "Mes-p severe":
            type_yolo = 2
        elif glo_pathology_type == "nodular sclerosis":
            type_yolo = 3
        elif glo_pathology_type == "sclerosis":
            type_yolo = 4
        else:
            type_yolo = 5

        out = open(
            result_path + glo_pathology_type + "_" + img[:-5] + "_" + str(data_level) + "_" + str(i) + "_" + ".txt",
            'w')
        out.write(
            str(type_yolo) + " " + str(x_center) + " " + str(y_center) + " " + str(w_yolo) + " " + str(h_yolo) + '\n')
        out.close()

def main(args_list, n_workers=4):
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(slide_process, arg) for arg in args_list]

        # 使用tqdm创建进度条
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
    return results

if __name__ == '__main__':
    path = "Z:/DN_Slide/"  # "E:/DN_Slide/"切片地址
    WSI_dir = os.listdir(path)
    slidelist = []
    for slide_name in WSI_dir:
        if slide_name[-5:] == ".mrxs" and slide_name[:2] == "18":  # 加判断只处理18年的
            slidelist.append(slide_name)
        elif slide_name[-5:] == ".mrxs" and slide_name[7:9] == "18":
            slidelist.append(slide_name)
    slidelist = slidelist[29:]
    args_list = [(img, path) for img in slidelist]
            # 执行
    main(args_list, n_workers=16)  # 使用32个工作进程


















    #
    #
    #
    # maskimg = Image.new('L', (m1, n1), 0)
    # for i in range(len(down_level_shape)):
    #     ImageDraw.Draw(maskimg).polygon(down_level_shape[i], outline=255, fill=255)
    # mask = numpy.array(maskimg)
    # io.imsave(img_path + ".png", mask)
    #
    # png_path = img_path + ".png"
    # maskslide = openslide.open_slide(png_path)
    # data_gen = DeepZoomGenerator(maskslide, tile_size=512, overlap=0, limit_bounds=False)
    #
    # level = data_gen.level_count
    # mkdir("E:\\result\\" + img + ".png" + "\\" + str(level) + "\\")
    # result_path = "E:\\result\\" + img + ".png" + "\\" + str(level) + "\\"
    # [i_range, j_range] = data_gen.level_tiles[level - 1]

    # for i in tqdm(range(i_range)):
    #     for j in range(j_range):
    #         tile_img = data_gen.get_tile(level - 1, (i, j))
    #         tile_array = np.array(tile_img)
    #         io.imsave(result_path + "/" + img[:-5] + str(level) + "_" + str(i) + "_" + str(j) + ".png", tile_array)  # 保存图像