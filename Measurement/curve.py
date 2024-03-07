import cv2
import numpy as np
from shapely.geometry import LineString, Polygon, LinearRing
from shapely import affinity
from shapely.ops import split
import shapely
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from shapely.affinity import translate
import numpy as np
import math
from readtxt import readcoords

image = np.zeros((2000, 2000, 3), dtype=np.uint8)
image.fill(255)

def perpendicular_lines(img, poly):
    # 将多边形顶点转换为适合cv2.fitLine的格式
    cv2.polylines(img, [poly], isClosed=True, color=(0, 0, 0), thickness=1)
    poly_pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))

    # 计算多边形的拟合直线
    [vx, vy, x, y] = cv2.fitLine(poly_pts, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((img.shape[1] - x) * vy / vx) + y)

    # 画出拟合直线
    cv2.line(img, (img.shape[1] - 1, righty), (0, lefty), (255, 0, 0), 1)

    # 计算并绘制垂线
    # 直线上点的间隔（50像素）
    step = 10
    length = 1000  # 控制垂线长度

    points = []
    polygon = Polygon(poly)
    # 沿直线方向计算点，并绘制垂线
    for t in np.arange(-500, 500, step / np.sqrt(vx ** 2 + vy ** 2)):
        pt_on_line = (x + t * vx, y + t * vy)
        perp_start = (int(pt_on_line[0] - vy * length), int(pt_on_line[1] + vx * length))
        perp_end = (int(pt_on_line[0] + vy * length), int(pt_on_line[1] - vx * length))
        #cv2.line(img, perp_start, perp_end, (0, 255, 0), 1)

        line_para = [perp_start, perp_end]
        line = LineString(line_para)
        # 计算交点
        intersection = polygon.intersection(line)

        if intersection.is_empty:
            # print("没有交点")
            length_intersection = 0
        else:
            # 交点可能是点也可能是多个点的集合
            if intersection.geom_type == 'Point':
                print(f"交点: {(intersection.x, intersection.y)}")
                length_intersection = 0
            elif intersection.geom_type == 'MultiPoint':
                print("多个交点:")
                length_intersection = 0
                for point in intersection:
                    print(f"({point.x}, {point.y})")
            elif intersection.geom_type == 'LineString' or intersection.geom_type == 'LinearRing':
                print("交点形成了一条线", f"长度:{intersection.length}")
                centerpoint = (int(intersection.centroid.x), int(intersection.centroid.y))

                for point in intersection.coords:
                    print(point)

            elif intersection.geom_type == 'MultiLineString':
                print(f"交点形成了{len(intersection.geoms)}条线", f"长度:{intersection.length}")

                centerpoint = (int(intersection.centroid.x), int(intersection.centroid.y))

                for crossline in intersection.geoms:
                    intersection_line_points = np.array(list(crossline.coords), dtype=np.int32)
                    for point in crossline.coords:
                        print(point)
            else:
                print(f"交点类型: {intersection.geom_type}")
                print(f"长度:{intersection.length}")
            points.append(centerpoint)
    return points

def fitcurve(points):
    sorted_points = sorted(points, key=lambda point: point[0])
    # 分解数据点为两个列表：x和y
    x, y = zip(*sorted_points)  # 这会创建两个元组

    # 将元组转换为NumPy数组以便于处理
    x = np.array(x)
    y = np.array(y)

    # 使用numpy的polyfit进行多项式拟合，这里拟合一个二次多项式
    coefficients = np.polyfit(x, y, 5)

    # 根据拟合得到的系数生成多项式函数
    polynomial = np.poly1d(coefficients)

    # 生成用于绘制的数据
    x_fit = np.linspace(x[0], x[-1], 50)
    y_fit = polynomial(x_fit)

    # 绘制原始数据点和拟合得到的曲线
    plt.plot(x, y, 'o', label='Original data', markersize=10)
    plt.plot(x_fit, y_fit, '-', label='Fitted curve')
    plt.legend()
    plt.show()

    # 将曲线点转换为适合绘图的坐标形式
    # 注意：你可能需要根据你的图像尺寸和数据点范围调整缩放和偏移量
    points = np.array([x_fit, y_fit]).T.reshape(-1, 1, 2)
    points = np.int32(points)

    # 绘制曲线
    cv2.polylines(image, [points], isClosed=False, color=(0, 0, 255), thickness=2)

def calculate_rotation_angle(line):
    [(x1, y1), (x2, y2)] = line
    # 计算斜率
    m = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')

    # 计算与x轴的角度（弧度）
    theta = math.atan(m)

    # 转换为度
    angle = math.degrees(theta)

    # 返回角度，使之平行于x轴
    return -theta
def fitline(img,poly):
    # 将多边形顶点转换为适合cv2.fitLine的格式
    poly_pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))

    # 计算多边形的拟合直线
    [vx, vy, x, y] = cv2.fitLine(poly_pts, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((img.shape[1] - x) * vy / vx) + y)

    line = [(img.shape[1] - 1, righty), (0, lefty)]
    return line

def coords_to_curve(polygon_coords):


    rad = calculate_rotation_angle(fitline(image, np.array(polygon_coords, dtype=np.int32)))

    poly = np.array(polygon_coords, dtype=np.int32)
    polygon = Polygon(poly)
    polygon = affinity.rotate(polygon, rad + 0.04, origin='centroid', use_radians=True)
    poly = np.asarray(polygon.exterior.coords, dtype=np.int32)

    points = perpendicular_lines(image, poly)

    fitcurve(points)

    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# image = np.zeros((2000, 2000,3), dtype=np.uint8)
# image.fill(255)
# # 定义多边形的顶点坐标
# filelist = ['466','1037','1545']
# for coordfile in filelist:
#     polygon_coords = readcoords(f"H:/pyglcv/txt/contour_points_{coordfile}.txt")
#
#     # polygon_coords = [(50,150),(50,50), (80, 30), (200, 100), (130, 150), (400,200), (100, 200)]
#     # 创建多边形对象
#     coords_to_curve(polygon_coords)
