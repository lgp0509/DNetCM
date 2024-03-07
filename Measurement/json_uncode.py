import os
from tqdm.autonotebook import tqdm
from math import ceil
import numpy
from PIL import ImageFont, ImageDraw, Image
import geojson
import shapely
from shapely.geometry import shape
from shapely.strtree import STRtree
from shapely.geometry import Point
from shapely.geometry import Polygon
import cv2
import numpy as np
import ms
from datatoexcel import lenthlist_to_excel
import shapely
import curve
import os
from imgprep import color_normalize

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import openslide
from openslide.deepzoom import DeepZoomGenerator
import warnings
warnings.filterwarnings('ignore')

def cv_show(img,name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def modif_list(list,filename):
    if "norm" in filename:
        # 对列表中的每个元素减去10
        modified_list = [x - 10 for x in list]
        # 删除计算后小于10的数据
        list = [x for x in modified_list if x >= 10]
    return list
def filter_whitefig(imgA,poly,white_threshold = 200, rate = 0.5):
    from shapely.geometry import Polygon
    if poly.area == 0:
        return Polygon()
    # 假设imgA是你的OpenCV图像，poly是Shapely多边形
    # imgA = cv2.imread('your_image_path.jpg')  # 读取图像
    # poly = Polygon([(x1, y1), (x2, y2), ...])  # 定义多边形
    # 将Shapely多边形转换为OpenCV点集
    poly_points = np.array(poly.exterior.coords).reshape((-1, 1, 2)).astype(np.int32)

    # 创建一个与imgA相同大小的全黑掩模
    mask = np.zeros(imgA.shape[:2], np.uint8)

    # 在掩模上绘制多边形（白色）
    cv2.fillPoly(mask, [poly_points], 255)

    # 应用掩模以提取多边形区域
    masked_img = cv2.bitwise_and(imgA, imgA, mask=mask)

    # 计算直方图，这里以灰度图像为例
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], mask, [256], [0, 256])

    # 检测偏白区域，设置阈值为200
    total_pixels = np.sum(hist)
    white_pixels = np.sum(hist[white_threshold:])

    # 如果偏白的像素超过了总像素的一定比例（例如30%），则认为区域偏白
    if white_pixels / total_pixels > rate:
        poly = Polygon()
        print("偏白")
    else:
        print("不偏白")
    return poly
def remove_outliers(data):
    if len(data) != 0:
        clean_data = []
        all_data = []
        for sublist in data:
            all_data.extend(sublist)
        q1, q3 = np.percentile(all_data, [25, 75])
        iqr = q3 - q1
        lower_bound = 0
        upper_bound = q3 # + 1.5 * iqr


        for sublist in data:
            # q1, q3 = np.percentile(sublist, [25, 75])
            # iqr = q3 - q1
            # lower_bound = q1 - 1.5 * iqr
            # upper_bound = q3 + 1.5 * iqr
            filtered_sublist = [x for x in sublist if x >= lower_bound and x <= upper_bound]
            clean_data.append(filtered_sublist)
        return clean_data
    else:
        return data

def mes(filepath, filename,polys, type,result):
    if type =="mes":
        color = (255, 0, 0)
    elif type =="cap":
        color = (0,0,255)
    else:
        color = (0,0,0)


    img = cv2.imread(f"{filepath}/{filename}")
    print(f"{filepath}/{filename}")
    # 获取图像的高度和宽度
    # height, width = img.shape[:2]
    # result = np.ones((height, width, 3), np.uint8) * 255  # 255表示白色

    length_lists = []
    j = 0
    for i, poly in enumerate(polys):
        if poly.geom_type == 'MultiPolygon':
            for poly_split in poly.geoms:
                polygon_array = np.array(poly_split.exterior.coords)
                proc_poly = ms.nparr_to_Polygon(polygon_array)
        else:
            polygon_array = np.asarray(poly.exterior.coords, dtype=np.int32)
            proc_poly = ms.nparr_to_Polygon(polygon_array)

        if proc_poly.geom_type == 'MultiPolygon':
            for poly_split in proc_poly.geoms:
                # poly_split = filter_whitefig(img, poly_split)
                if poly_split.area > 300:
                    lengthlist, result = ms.draw_perpendicular_lines_and_measure(poly_split, result)
                    poly_split = np.asarray(poly_split.exterior.coords, dtype=np.int32)
                    cv2.polylines(result, [poly_split], isClosed=True, color = color, thickness=3)
                    if lengthlist != 0:
                        # lengthlist = modif_list(lengthlist, filename)
                        length_lists.append(lengthlist)
                        # 在图像上添加轮廓的索引标签，写数字
                        x, y = poly_split[0]  # 获取轮廓的起始点坐标
                        cv2.putText(result, str(j), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        j = j + 1
        else:
            # poly = filter_whitefig(img, poly)
            if proc_poly.area != 0:
                lengthlist, result = ms.draw_perpendicular_lines_and_measure(proc_poly, result)
                proc_poly = np.asarray(proc_poly.exterior.coords, dtype=np.int32)
                cv2.polylines(result, [proc_poly], isClosed=True, color = color, thickness=3)
                if lengthlist != 0:
                    # lengthlist = modif_list(lengthlist, filename)
                    length_lists.append(lengthlist)
                    # 在图像上添加轮廓的索引标签，写数字
                    x, y = proc_poly[0]  # 获取轮廓的起始点坐标
                    cv2.putText(result, str(j), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    j = j + 1
    lenthlist_to_excel(remove_outliers(length_lists),filepath,filename+type)

    # 显示结果
    combined_image = cv2.addWeighted(result, 0.3, img, 0.7, 1.0)
    # cv_show(combined_image, "result")
    return  combined_image

path = r'H:\pyglcv\figglo'
# img = r'Mes-p mild_170721 PAS_0_1_.jpg'
# img = r'Mes-p severe_162676 PAS_0_2_.jpg'
img = r'Normal glo_172980 PAS_0_13_.jpg'
json_fname = path + "/" + img + ".json"  # input geojson file

image = cv2.imread(f"{path}/{img}")
# 获取图像的高度和宽度
height, width = image.shape[:2]
result = np.ones((height, width, 3), np.uint8) * 255  # 255表示白色


with open(json_fname) as f:
    allobjects = geojson.load(f)
print("done loading")
allshapes = []
for obj in allobjects:
    if "cap" in obj.properties.get('classification').get('name'):
        allshapes.append(shape(obj["geometry"]))
print(len(allshapes))
print("done converting")
mes(path,img,allshapes,"cap",result)

with open(json_fname) as f:
    allobjects = geojson.load(f)
print("done loading")
allshapes = []
for obj in allobjects:
    if "mes" in obj.properties.get('classification').get('name'):
        allshapes.append(shape(obj["geometry"]))
print(len(allshapes))
print("done converting")
comboimage = mes(path,img,allshapes,"mes",result)

#region 比例尺
# 比例尺长度（以像素为单位）
scale_length_px = 100
# 比例尺长度（以μm为单位）
scale_length_um = scale_length_px * 0.13
# 比例尺起始点坐标
start_point = (width-300, height-200)
# 比例尺结束点坐标
end_point = (start_point[0] + scale_length_px, start_point[1])
# 绘制比例尺线段
cv2.line(comboimage, start_point, end_point, (0, 0, 0), thickness=2)

# 保存图像
cv2.imwrite(f'H:/pyglcv/output/{img}',comboimage)

# 使用PIL加载图像
pil_img = Image.open(f'H:/pyglcv/output/{img}')

# 获取PIL图像的绘图对象
draw = ImageDraw.Draw(pil_img)

# 加载自定义字体文件
font = ImageFont.truetype(r'C:\Windows\Fonts\arialbd.ttf', 20)

# 在比例尺上方绘制文字
text = f"{int(scale_length_um)} μm"
draw.text((start_point[0], start_point[1] - 25), text, fill=(0, 0, 0), font=font)

# 保存带有文本的图像
pil_img.save(f'H:/pyglcv/output/{img}')

# 显示图像
pil_img.show()
#endregion

