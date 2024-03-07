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

from skimage import io, color, data, measure
import numpy as np
import matplotlib.pyplot as plt
import openslide
from openslide.deepzoom import DeepZoomGenerator
import warnings
warnings.filterwarnings('ignore')

path = "Y://RESULT//unet_output//100"
dir = os.listdir(path)
slideset = set()
#读tiles
for silde_name in dir:
    slideset.add(silde_name[:12])
    slidelist = sorted(list(slideset))
for slide in slidelist:
    png_path = "E://DN_Slide//" + slide[:-2] + ".mrxs.png"
    maskslide = openslide.open_slide(png_path)
    data_gen = DeepZoomGenerator(maskslide, tile_size=512, overlap=0, limit_bounds=False)
    level = data_gen.level_count
    [i_range, j_range] = data_gen.level_tiles[level - 1]
    for i in tqdm(range(i_range)):
        for j in range(j_range):
            tile_filename = str(slide) + "_" + str(i) + "_" + str(j) + ".png"
            mask_path = path + "//" + tile_filename
            if tile_filename in dir:
                tile = np.array(io.imread(mask_path))
            else:
                tile = np.array(color.rgb2gray(io.imread("Y://RESULT//Slide_tiles.mask//" + tile_filename)))
            if j == 0:
                axisyarr = tile
            else:
                axisyarr = np.concatenate((axisyarr,tile),axis=0)
        if i == 0:
            fullimg_arr = axisyarr
        else:
            fullimg_arr = np.concatenate((fullimg_arr,axisyarr),axis=1)
    arr = fullimg_arr
    io.imsave(path[:11] + "output_png" + "//" + slide[:-2] +".mrxs.png",fullimg_arr)

