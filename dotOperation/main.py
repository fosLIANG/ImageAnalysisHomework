import cv2 as cv
import numpy as np
import math


def cv_show(name, img):
    '''
     显示图像
    '''
    cv.imshow(name, img)
    cv.waitKey()
    cv.destroyAllWindows()


def show_two_pictures(img_one, img_two):
    '''
     对比显示两张图片
    '''
    cv_show("Two Pictures", np.hstack((img_one, img_two)))


def log_img(img, c):
    '''
     对数变换
    '''
    (h, w, c) = img.shape
    img_copy = np.zeros((h, w, c), np.float32)
    for i in range(h):
        for j in range(w):
            img_copy[i, j, 0] = c * math.log(1.0 + img[i, j, 0])
            img_copy[i, j, 1] = c * math.log(1.0 + img[i, j, 1])
            img_copy[i, j, 2] = c * math.log(1.0 + img[i, j, 2])
    cv.normalize(img_copy, img_copy, 0, 255, cv.NORM_MINMAX)
    # 使用normalize后一般数据都为浮点型数据
    # opencv中的图像数据类型为uchar
    # 因此 一般若是对图像进行归一化处理后都需要后面紧跟convertScaleAbs()进程转换
    img_copy = cv.convertScaleAbs(img_copy)
    return img_copy


def linear_extension(img):
    '''
     线性扩展
    '''
    (h, w) = img.shape
    flat_img = img.reshape((h * w,)).tolist()
    img_max = max(flat_img)
    img_min = min(flat_img)
    new_img = np.uint8(255 / (img_max - img_min) * (img - img_min))

    return new_img


def segmented_linear(img):
    '''
     分段线性变换
    '''
    (h, w) = img.shape
    flat_img = img.reshape(h * w, ).tolist()
    Mf = max(flat_img)
    Ms = min(flat_img)
    a = Ms + 1 / 3 * (Mf - Ms)
    b = a + 1 / 3 * (Mf - Ms)
    c = a / 2
    d = c + 2 * (b - a)
    Mg = d + 0.5 * (Mf - b)
    img_copy = np.zeros((h, w), np.float32)
    for i in range(h):
        for j in range(w):
            if img[i, j] <= a:
                img_copy[i, j] = 0.5 * img[i, j]
            elif a < img[i, j] <= b:
                img_copy[i, j] = c + (d - c) / (b - a) * (img[i, j] - a) + c
            else:
                img_copy[i, j] = (Mg - d) / (Mf - b) * (img[i, j] - b) + d
    cv.normalize(img_copy, img_copy, 0, 255, cv.NORM_MINMAX)
    img_copy = cv.convertScaleAbs(img_copy)
    return img_copy


img = cv.resize(cv.imread("C:/Users/Jun/PycharmProjects/image/4.jpg", 0), (400, 250))
show_two_pictures(img, linear_extension(img))

img = cv.resize(cv.imread("C:/Users/Jun/PycharmProjects/image/3.jpg", 0), (400, 250))
show_two_pictures(img, segmented_linear(img))

img = cv.resize(cv.imread("C:/Users/Jun/PycharmProjects/image/3.jpg"), (400, 250))
show_two_pictures(img, log_img(img, 1.0))
