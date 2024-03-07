import cv2
import os
import numpy as np
from tqdm.autonotebook import tqdm

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

def eq_hist_clahe(img, clipL = 2.0, tilesz = (50,50)):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    # 分离L、A、B通道
    channels = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=clipL, tileGridSize=tilesz)
    # 对每个通道应用直方图均衡化
    clahe.apply(channels[0], channels[0])
    ycrcb = cv2.merge(channels)

    # 将新的LAB图像转换回BGR颜色空间
    equalized_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return equalized_image

def color_normalize(src_image, ref_image_path = ""):
    if ref_image_path != "":
        ref_image = cv2.imread(ref_image_path)
        ref_lab = cv2.cvtColor(ref_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_mean, ref_stddev = cv2.meanStdDev(ref_lab)
        ref_mean = ref_mean.reshape(1, 1, 3)
        ref_stddev = ref_stddev.reshape(1, 1, 3)
    else:
        ref_mean = np.array([208, 147, 114]).reshape(1, 1, 3)
        ref_stddev = np.array([33, 15, 10]).reshape(1, 1, 3)

    # 转换到LAB颜色空间
    src_lab = cv2.cvtColor(src_image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 计算均值和标准差
    src_mean, src_stddev = cv2.meanStdDev(src_lab)



    # 将均值和标准差调整为正确的形状
    src_mean = src_mean.reshape(1, 1, 3)
    src_stddev = src_stddev.reshape(1, 1, 3)


    # 应用标准化公式
    src_lab_normalized = ((src_lab - src_mean) / (src_stddev + 1e-10)) * ref_stddev + ref_mean

    # 确保值在正确的范围内
    src_lab_normalized = np.clip(src_lab_normalized, 0, 255)

    # 转换回BGR颜色空间
    src_normalized_bgr = cv2.cvtColor(src_lab_normalized.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return src_normalized_bgr

def eq_hist_full(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # 分离L、A、B通道
    channels = cv2.split(ycrcb)
    # 对每个通道应用直方图均衡化
    cv2.equalizeHist(channels[0], channels[0])
    ycrcb = cv2.merge(channels)

    # 将新的LAB图像转换回BGR颜色空间
    equalized_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return equalized_image

# for type in ["train","val","test"]:
#     path = f'C:/yolov9/datasets/glo2/images/{type}/'  # "E:/DN_Slide/"切片地址
#     dir = os.listdir(path)
#     for patch in tqdm(dir):
#         # 定义原始文本文件路径
#         file_path = path + patch
#         image = cv2.imread(file_path)
#
#         equalized_image = eq_hist_clahe(image, clipL=1.0,tilesz=(100,100))
#         equalized_image = color_normalize(equalized_image)
#
#         mkdir(path + "hist/")
#         cv2.imwrite(path + "hist/" + patch,equalized_image)

from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import os

def process_image(args):
    patch, path = args
    file_path = os.path.join(path, patch)
    image = cv2.imread(file_path)

    # 假设 eq_hist_clahe 和 color_normalize 是已定义的函数
    equalized_image = eq_hist_clahe(image, clipL=1.0, tilesz=(100,100))
    equalized_image = color_normalize(equalized_image)

    hist_dir = os.path.join(path, "hist")
    os.makedirs(hist_dir, exist_ok=True)
    cv2.imwrite(os.path.join(hist_dir, patch), equalized_image)
    # print(patch)

def main(args_list, n_workers=4):
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_image, arg) for arg in args_list]

        # 使用tqdm创建进度条
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
    return results

if __name__ == '__main__':
    for type in ["train", "val", "test"]:
        path = f'C:/yolov9/datasets/glo2/images/{type}/'  # 调整为您的路径
        dir = os.listdir(path)
        args_list = [(patch, path) for patch in dir if patch[-3:]=="jpg"]

        # 执行
        main(args_list, n_workers=16)  # 使用32个工作进程
