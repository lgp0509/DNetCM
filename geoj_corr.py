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

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import openslide
from openslide.deepzoom import DeepZoomGenerator
import warnings
warnings.filterwarnings('ignore')


def type_get(i,allobjects):
    glo_pathology_type = allobjects[i].get('properties')
    if 'classification' in glo_pathology_type:
        glo_pathology_type = glo_pathology_type.get('classification').get('name')
    else:
        glo_pathology_type = 'notype'
    return glo_pathology_type

path = "Z:/DN_Slide/"
WSI_dir = os.listdir(path)
slidelist = []
for slide_name in WSI_dir:
    if slide_name[-5:] == ".mrxs" and slide_name[:2] =="18": #加判断只处理18年的
        slidelist.append(slide_name)
    elif slide_name[-5:] == ".mrxs" and slide_name[7:9] =="18":
        slidelist.append(slide_name)



for img in slidelist:
    reform_objects = []
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

    reform_shapes = []
    for i, polyitem in enumerate(allshapes):
        type_name = type_get(i, allobjects)
        if polyitem.geom_type == 'MultiPolygon':
            # polyitem.buffer(-2)
            # polyitem.buffer(2)
            for item in polyitem:
                if item.area > 5000:
                    reform_shapes_list = [list(point) for point in list(item.exterior.coords)]
                    p = reform_shapes_list
                    test_dict = {
                        "type": "Feature",
                        "id": "PathAnnotationObject",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [p]
                        },
                        "properties": {
                            "classification": {
                                "name": type_name,
                                "colorRGB": -6370472
                            },
                            "measurements": []
                        }
                    }
                    reform_objects.append(test_dict)
        elif polyitem.geom_type == 'Polygon':
            reform_shapes_list = [list(point) for point in list(polyitem.exterior.coords)]
            p = reform_shapes_list
            test_dict = {
                "type": "Feature",
                "id": "PathAnnotationObject",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [p]
                },
                "properties": {
                    "classification": {
                        "name": type_name,
                        "colorRGB": -6370472
                    },
                    "measurements": []
                }
            }
            reform_objects.append(test_dict)
    with open(json_fname, 'w') as outfile:
        geojson.dump(reform_objects, outfile)
