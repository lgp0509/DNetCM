import cv2
import numpy as np
from shapely.geometry import LineString, Polygon, LinearRing
from shapely import affinity
import shapely.geometry
from readtxt import readcoords

image = np.zeros((1280, 1280, 3), dtype=np.uint8)
image.fill(255)

def tempreadpic():
    # 二值法
    img = cv2.imread(r"H:\pyglcv\output\n.jpg")

    # 获取图像的高度和宽度
    height, width = img.shape[:2]
    scale_rate = 0.65

    size_height = int(height * scale_rate)
    size_width = int(width * scale_rate)

    # 计算要截取的区域的左上角坐标
    top_left_x = (width - size_width) // 2
    top_left_y = (height - size_height) // 2

    # 截取中心的500x500像素图像
    center_crop = img[top_left_y:top_left_y + size_height, top_left_x:top_left_x + size_width]
    return center_crop
def print_poly(poly):
    tmpslide = tempreadpic()
    fig = np.zeros(tmpslide.shape, dtype=np.uint8)
    fig.fill(255)
    poly = np.asarray(poly.exterior.coords, dtype=np.int32)
    cv2.polylines(fig, [poly], isClosed=True, color=(255, 0, 0), thickness=3)
    combined_image = cv2.addWeighted(fig, 0.5, tmpslide, 0.5, 1.0)
    cv2.imshow('Image with Fit Line and Perpendiculars', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def stats(data):
    # 计算均值
    mean_value = np.around(np.mean(data),2)

    # 计算标准差
    std_deviation = np.around(np.std(data),2)

    # 计算四分位数
    q1 = np.around(np.quantile(data, 0.25),2)
    m = np.around(np.median(data),2)
    q3 = np.around(np.quantile(data, 0.75),2)
    print(f"Mean±SD:{mean_value}±{std_deviation}, Median:{m}({q1},{q3}), n = {len(data)}")
    statscon = f"Mean, SD:{mean_value}, {std_deviation}, Median:{m}({q1},{q3}), n = {len(data)}"
    return statscon
def intersectioncalc(poly ,line_params, resultimg):
    [point1 , point2] = line_params
    polygon = Polygon(poly)
    if not polygon.is_valid:
        # 尝试修复多边形
        polygon = polygon.buffer(0)
        # 或者，如果可用，使用make_valid
        # poly = make_valid(poly)

    # 创建直线对象
    line = LineString([point1 , point2])

    # 计算交点
    intersection = polygon.intersection(line)
    length_intersection = []
    # 输出交点信息
    if intersection.is_empty:
        # print("没有交点")
        length_intersection.append(0)
    else:
        # 交点可能是点也可能是多个点的集合
        if intersection.geom_type == 'Point':
            print(f"交点: {(intersection.x, intersection.y)}")
            length_intersection.append(0)
        elif intersection.geom_type == 'MultiPoint':
            print("多个交点:")
            length_intersection.append(0)
            for point in intersection:
                print(f"({point.x}, {point.y})")
        elif intersection.geom_type == 'LineString' or intersection.geom_type == 'LinearRing':
            # print("交点形成了一条线",f"长度:{intersection.length}")
            length_intersection.append(intersection.length)

            intersection_line_points = np.array(list(intersection.coords), dtype=np.int32)
            cv2.polylines(resultimg, [intersection_line_points], False,  color= (0, 0, 255), thickness=1)
            # cv2.putText(resultimg, str(np.around(intersection.length,2)), org = intersection_line_points[-1], color=(0, 0, 0), thickness=1, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5)

            # for point in intersection.coords:
            #     print(point)
        elif intersection.geom_type == 'MultiLineString':
            # print(f"交点形成了{len(intersection.geoms)}条线",f"长度:{intersection.length}")
            for crossline in intersection.geoms:
                intersection_line_points = np.array(list(crossline.coords), dtype=np.int32)
                cv2.polylines(resultimg, [intersection_line_points], False, color=(0, 0, 255), thickness=1)
                length_intersection.append(crossline.length)
                # for point in crossline.coords:
                #     print(point)
            # cv2.putText(resultimg, str(np.around(intersection.length,2)), org=intersection_line_points[-1], color=(0, 0, 0), thickness=1,
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5)
        else:
            print(f"交点类型: {intersection.geom_type}")
            print(f"长度:{intersection.length}")
            length_intersection.append(intersection.length)
    return length_intersection, resultimg
def draw_perpendicular_lines_and_measure(poly,resultimg):
    if isinstance(poly, shapely.geometry.base.BaseGeometry):
        # Do something if poly is a Shapely geometry
        poly = np.asarray(poly.exterior.coords, dtype=np.int32)

    if len(poly) < 3:
        return 0

    cv2.polylines(image, [poly], isClosed=True, color=(0, 0, 0), thickness=3)
    poly_pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))# 将多边形顶点转换为适合cv2.fitLine的格式



    # 计算多边形的拟合直线
    [vx, vy, x, y] = cv2.fitLine(poly_pts, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((image.shape[1] - x) * vy / vx) + y)

    # 画出拟合直线
    cv2.line(image, (image.shape[1] - 1, righty), (0, lefty), (255, 0, 0), 2)


    # 计算并绘制垂线
    # 直线上点的间隔（50像素）
    step = 5
    length = 2000  # 控制垂线长度

    lengthlist = []
    # 沿直线方向计算点，并绘制垂线
    for t in np.arange(-1000, 1000, step / np.sqrt(vx ** 2 + vy ** 2)):
        pt_on_line = (x + t * vx, y + t * vy)
        perp_start = (int(pt_on_line[0] - vy * length), int(pt_on_line[1] + vx * length))
        perp_end = (int(pt_on_line[0] + vy * length), int(pt_on_line[1] - vx * length))
        #cv2.line(image, perp_start, perp_end, (0, 255, 0), 1)

        line_para = [perp_start, perp_end]

        length_lines, resultimg = intersectioncalc(poly, line_para, resultimg)
        for length_line in length_lines:
            if length_line > 10:
                lengthlist.append(length_line)


    statscon = stats(lengthlist)
    cv2.putText(image, statscon, org=(0,20), color=(0, 0, 0),
                thickness=1,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5)

    # cv2.imshow('Image with Fit Line and Perpendiculars', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return lengthlist, resultimg

def lengthlist_calc(polys):
    for polygon_coords in polys:

        # 创建一个空白图像
        image = np.zeros((2000, 2000, 3), dtype=np.uint8)
        image.fill(255)

        # 创建多边形对象
        poly = np.array(polygon_coords, dtype=np.int32)
        polygon = Polygon(poly)
        polygon = affinity.rotate(polygon, 0, origin='centroid')
        # polygon = polygon.simplify(10, preserve_topology=False)
        poly = np.asarray(polygon.exterior.coords, dtype=np.int32)

        lengthlist = draw_perpendicular_lines_and_measure(image, poly)
        print(lengthlist)

        # cv2.imshow('Image with Fit Line and Perpendiculars', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



def nparr_to_Polygon(polygon_coords, rot_rad = 0,bufferlvl = 4):
    proc_polygon = Polygon()
    poly = np.array(polygon_coords, dtype=np.int32)
    polygon = Polygon(poly)
    polygon = affinity.rotate(polygon, rot_rad, origin='centroid')
    print(polygon.geom_type,'化简前',polygon.area)
    if polygon.area > 0:
        polygon = polygon.buffer(-2)
        polygon = polygon.buffer(2)
        # polygon = polygon.simplify(2, preserve_topology=False)
        polygon = polygon.buffer(-bufferlvl)
        polygon = polygon.buffer(bufferlvl)
        if polygon.geom_type == 'MultiPolygon':
            for subpoly in polygon.geoms:
                if subpoly.area > 2000 and subpoly.geom_type ==  'Polygon':
                    proc_polygon = proc_polygon.union(subpoly)
        else:
            proc_polygon = polygon
    else:
        proc_polygon = polygon
    print('化简后',proc_polygon.area)
    return proc_polygon

