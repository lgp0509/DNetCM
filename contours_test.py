import os
from tqdm.autonotebook import tqdm
from math import ceil
import numpy
from PIL import Image, ImageDraw
import geojson
import shapely
from shapely.geometry import shape
from shapely.geometry import asShape
from shapely.strtree import STRtree
from shapely.geometry import Point
from shapely.geometry import Polygon
from skimage import io
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

path = "Y://RESULT//output_png//"
dir = os.listdir(path)
slidelist = []
for slidename in dir:
    if slidename[-4:] == ".png":
        slidelist.append(slidename)

for slide_name in tqdm(slidelist):
    arr =  io.imread(path + slide_name)
    contours = measure.find_contours(arr, 150)
    allshapes = []
    allobjects = []
    for i in range(len(contours)):
        polygenxy = contours[i]
        polygenpointlist = []
        for item in polygenxy:
            a = item.tolist()
            polygenpointlist.append([a[1] * 8, a[0] * 8])
        p = polygenpointlist
        if Polygon(p).area > 1000:
            test_dict = {
                "type": "Feature",
                "id": "PathAnnotationObject",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [p]
                },
                "properties": {
                    "classification": {
                        "name": "glo",
                        "colorRGB": -6370472
                    },
                    "measurements": []
                }
            }
            allobjects.append(test_dict)
    d = path + slide_name[:-3] + ".json"
    with open(path + slide_name[:-3] + ".json", 'w') as outfile:
        geojson.dump(allobjects, outfile)