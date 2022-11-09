import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import math
import cv2


# 运动处理
def motion_process(slp, angle):
    gt = np.zeros(slp)
    center = (slp[0] - 1) / 2  # 图像中心
    tan = math.tan(angle * math.pi / 180)
    for i in range(20):
        offset = round(i * tan)
        gt[int(center + offset), int(center - offset)] = 1
    return gt / gt.sum()


# 模糊处理
def make_fuzzy(input, gt, eps):
    input_fft = fft.fft2(input)  # 傅里叶变换
    gt_fft = fft.fft2(gt) + eps
    fuzzy = fft.ifft2(input_fft * gt_fft)
    fuzzy = np.abs(fft.fftshift(fuzzy))
    return fuzzy


# 维纳滤波
def wiener(noisy_input, gt, eps, K=0.01):
    input_fft = fft.fft2(noisy_input)
    gt_fft = fft.fft2(gt) + eps
    gt_fft_1 = np.conj(gt_fft) / (np.abs(gt_fft) ** 2 + K)
    res = fft.ifft2(input_fft * gt_fft_1)
    res = np.abs(fft.fftshift(res))
    return res


img = cv2.imread('C:/Users/Jun/PycharmProjects/Wiener/picture/1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换成灰度图片
[h, w] = img.shape
plt.figure(1)
plt.subplot(131)
plt.xlabel("Origin gray Image")
plt.gray()
plt.imshow(img)  # 显示原图像
# 进行运动模糊处理45 degree
gt = motion_process((h, w), 45)
fuzzy = np.abs(make_fuzzy(img, gt, 1e-3))
fuzzy_noisy = fuzzy + 0.1 * fuzzy.std()
np.random.standard_normal(fuzzy.shape)  # 添加噪声
plt.subplot(132)
plt.xlabel("motion & fuzzy & noisy")
plt.imshow(fuzzy_noisy)  # 显示运动模糊且添加噪声的图像
result = wiener(fuzzy_noisy, gt, 0.1 + 1e-3)  # 对该图像进行维纳滤波
plt.subplot(133)
plt.xlabel("wiener(k=0.01)")
plt.imshow(result)
plt.show()
