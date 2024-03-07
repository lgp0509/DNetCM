import pandas as pd
import numpy as np
import os
from shapely.geometry import LineString, Polygon, LinearRing
from shapely import affinity
from ms import draw_perpendicular_lines_and_measure
from readtxt import readcoords


# polys =[]
#
# filelist = ['466','1037','1545']
# for coordfile in filelist:
#     polygon_coords = readcoords(f"H:/pyglcv/txt/contour_points_{coordfile}.txt")
#     polys.append(polygon_coords)
#
# # 创建空的 DataFrame
# df = pd.DataFrame()
#
# # 用于存储所有长度列表的临时容器
# length_lists = []
#
# # 先计算所有长度列表，以找出最大长度
# for polygon_coords in polys:
#     image = np.zeros((500, 500, 3), dtype=np.uint8)
#     image.fill(255)
#     poly = np.array(polygon_coords, dtype=np.int32)
#     polygon = Polygon(poly)
#     polygon = affinity.rotate(polygon, 25, origin='centroid')
#     # polygon = polygon.simplify(10, preserve_topology=False)
#     poly = np.asarray(polygon.exterior.coords, dtype=np.int32)
#     lengthlist = draw_perpendicular_lines_and_measure(poly)
#     length_lists.append(lengthlist)

def lenthlist_to_excel(length_lists,directory_path,fname = "output"):
    # 计算最大长度

    # Extracting the part after the last backslash
    directory_path = directory_path.split("\\")[-1]


    df = pd.DataFrame()
    if len(length_lists) != 0:
        max_length = max(len(lst) for lst in length_lists)

        # 填充长度列表并添加到 DataFrame
        for i, lengthlist in enumerate(length_lists):
            # 如果长度列表短于最大长度，则用 NaN 填充至最大长度
            lengthlist += [np.nan] * (max_length - len(lengthlist))
            df[f"mes{i}"] = lengthlist

        # 保存到 Excel，忽略索引
        # 拼接目标目录路径
        directory_path
        target_directory = os.path.join("H:\\pyglcv\\mes", directory_path)

        # 如果目标目录不存在，则创建它
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        df.to_excel(f'H:\pyglcv\mes\{directory_path}\{fname}.xlsx', index=True)