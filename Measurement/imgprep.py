import cv2
import numpy as np

def color_normalize(src_image):
    ref_image = cv2.imread(r'H:\pyglcv\result_glo\ref.jpg')
    # 转换到LAB颜色空间
    src_lab = cv2.cvtColor(src_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 计算均值和标准差
    src_mean, src_stddev = cv2.meanStdDev(src_lab)
    ref_mean, ref_stddev = cv2.meanStdDev(ref_lab)

    ref_mean = np.array([208,147,114]).reshape(1, 1, 3)
    ref_stddev = np.array([33,15,10]).reshape(1, 1, 3)

    # 将均值和标准差调整为正确的形状
    src_mean = src_mean.reshape(1, 1, 3)
    src_stddev = src_stddev.reshape(1, 1, 3)
    ref_mean = ref_mean.reshape(1, 1, 3)
    ref_stddev = ref_stddev.reshape(1, 1, 3)

    # 应用标准化公式
    src_lab_normalized = ((src_lab - src_mean) / (src_stddev + 1e-10)) * ref_stddev + ref_mean

    # 确保值在正确的范围内
    src_lab_normalized = np.clip(src_lab_normalized, 0, 255)

    # 转换回BGR颜色空间
    src_normalized_bgr = cv2.cvtColor(src_lab_normalized.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return src_normalized_bgr

# 读取参考图片和需要标准化的图片

src_image = cv2.imread(r'H:\pyglcv\msgroup\I\143120 PAS.mrxs\Mes-p mild_143120 PAS_0_3_.jpg')

# 进行颜色标准化
normalized_image = color_normalize(src_image)

cv2.imwrite(r'H:\pyglcv\output\n.jpg',normalized_image)
#显示结果

cv2.imshow('Original', src_image)
cv2.imshow('Normalized', normalized_image)
#res =np.hstack((src_image,normalized_image))
#cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

