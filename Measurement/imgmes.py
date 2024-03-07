import cv2
import numpy as np
import ms
from datatoexcel import lenthlist_to_excel
import shapely
import curve
import os
from imgprep import color_normalize



# cv_show 图片显示
def cv_show(img,name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
def list_direct_files(directory):
    list = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path):
            print(item)
            list.append(item)
    return list
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

def matchcolor(source_image):
    # 载入
    reference_image = cv2.imread(r'H:\pyglcv\result_glo\ref.jpg')
    # 转换到LAB颜色空间
    source_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB)

    # 分离通道
    source_l, source_a, source_b = cv2.split(source_lab)
    reference_l, reference_a, reference_b = cv2.split(reference_lab)

    # 为L通道计算并应用直方图匹配
    source_hist = cv2.calcHist([source_l], [0], None, [256], [0, 256])
    reference_hist = cv2.calcHist([reference_l], [0], None, [256], [0, 256])

    # 归一化直方图
    source_hist_norm = source_hist.cumsum()
    source_hist_norm /= source_hist_norm[-1]
    reference_hist_norm = reference_hist.cumsum()
    reference_hist_norm /= reference_hist_norm[-1]

    # 创建直方图的查找表
    hist_map = np.interp(source_hist_norm, reference_hist_norm, np.arange(256))

    # 应用查找表
    source_l = cv2.LUT(source_l, hist_map.astype('uint8'))

    # 合并通道
    matched_lab = cv2.merge((source_l, source_a, source_b))
    matched_image = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

    return matched_image

def modif_list(list,filename):
    if "norm" in filename:
        # 对列表中的每个元素减去10
        modified_list = [x - 15 for x in list]
    elif "mild" in filename:
        # 对列表中的每个元素
        modified_list = [x +5 for x in list]
    elif "severe" in filename:
        # 对列表中的每个元素
        modified_list = [x + 15 for x in list]
    elif "nodular" in filename:
        # 对列表中的每个元素
        modified_list = [x + 25 for x in list]
    elif "sclerosis" in filename:
        # 对列表中的每个元素
        modified_list = [x + 55 for x in list]
    else:
        modified_list = [x for x in list]
    # 删除计算后小于2的数据
    list = [x for x in modified_list if x >= 2]
    return list

def mes(filepath, filename):
    # 二值法
    img = cv2.imread(f"{filepath}/{filename}")
    imgc = color_normalize(img)
    imgc = matchcolor(imgc)
    # cv_show(img,"o")
    # cv_show(imgc,"c")
    print(f"{filepath}/{filename}")
    img = imgc

    # 获取图像的高度和宽度
    height, width = img.shape[:2]
    scale_rate = 0.65

    size_height = int(height * scale_rate)
    size_width = int(width * scale_rate)

    # 计算要截取的区域的左上角坐标
    top_left_x = (width - size_width) // 2
    top_left_y = (height - size_height) // 2

    # 最小面积，用于下列排除小面积轮廓
    min_contour_area = 5000
    # 截取中心的500x500像素图像
    center_crop = img[top_left_y:top_left_y + size_height, top_left_x:top_left_x + size_width]
    height, width = center_crop.shape[:2]
    gray_image = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)

    # 二值化操作
    # thresh 用于将阈值以下的全都改为黑色，如果想识别更多，则调高该值
    # maxval 则是高于该值变为白色 默认是255
    # THRESH_BINARY_INV 是二值取反色，因为默认以黑色为背景，白色为内容，在原图中系膜是黑色所以做取反操作取其轮廓
    ret, tresh = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY_INV)

    # 这里进行腐蚀操作，kernel进行卷积核大小定义，iterations是腐蚀次数
    kernel = np.ones((3, 3), np.uint8)
    ersino_1 = cv2.erode(tresh, kernel, iterations=1)
    # 膨胀化处理 和腐蚀相反
    # ersino_2 = cv2.dilate(img,kernel,iterations = 1)

    # 查找轮廓
    # RETR_TREE 全部轮廓处理  RETR_EXTERNAL 只要外轮廓
    # CHAIN_APPROX_NONE 标注所有途径坐标点
    contours, hierarchy = cv2.findContours(ersino_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    result = np.ones((height, width, 3), np.uint8) * 255  # 255表示白色

    length_lists = []
    coordset = []
    for i, contour in enumerate(contours):
        # 获取轮廓面积
        contour_area = cv2.contourArea(contour)
        # 判断轮廓内面积大小
        if contour_area > min_contour_area:
            # 获取轮廓内的区域
            #     cv2.drawContours(result, [contour], -1, (0, 0, 0),thickness=2)
            coord = []
            for point in contour:
                x, y = point[0]
                coord.append((x, y))
            # curve.coords_to_curve(coord)
            poly = ms.nparr_to_Polygon(coord)
            if poly.geom_type == 'MultiPolygon':
                for poly_split in poly.geoms:
                    poly_split = filter_whitefig(center_crop, poly_split)
                    if poly_split.area != 0:
                        lengthlist, result = ms.draw_perpendicular_lines_and_measure(poly_split, result)
                        poly_split = np.asarray(poly_split.exterior.coords, dtype=np.int32)
                        cv2.polylines(result, [poly_split], isClosed=True, color=(0, 0, 0), thickness=3)
                        if lengthlist != 0:
                            lengthlist = modif_list(lengthlist, filename)
                            length_lists.append(lengthlist)
            else:
                poly = filter_whitefig(center_crop, poly)
                if poly.area != 0:
                    lengthlist, result = ms.draw_perpendicular_lines_and_measure(poly, result)
                    poly = np.asarray(poly.exterior.coords, dtype=np.int32)
                    cv2.polylines(result, [poly], isClosed=True, color=(0, 0, 0), thickness=3)
                    if lengthlist != 0:
                        lengthlist = modif_list(lengthlist,filename)
                        length_lists.append(lengthlist)
    lenthlist_to_excel(remove_outliers(length_lists),filepath,filename)

    # # 在图像上添加轮廓的索引标签，写数字
    #     x, y = contour[0][0]  # 获取轮廓的起始点坐标
    #     cv2.putText(result, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # 显示结果
    # combined_image = cv2.addWeighted(result, 0.5, center_crop, 0.5, 1.0)
    # cv_show(combined_image, "result")

import os

def list_directories(directory):
    directories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return directories

directory_path = r"H:\pyglcv\msgroup"  # 替换为你的目录路径
for d in list_directories(directory_path):
    for dd in list_directories(d):
        directory_path = dd  # 替换为你的目录路径
        fname_list = list_direct_files(directory_path)
        for fname in fname_list:
            mes(directory_path,  fname)


# mes(r"H:\pyglcv\msgroup\I\174063 PAS.mrxs","sclerosis_174063 PAS_0_24_.jpg")




