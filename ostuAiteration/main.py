import cv2
import numpy as np
import matplotlib as plt


def OSTU(grayImg):
    N = grayImg.size
    P = np.array([0] * 256)
    for i in range(grayImg.shape[0]):
        for j in range(grayImg.shape[1]):
            P[grayImg[i, j]] += 1
    P = P / N

    G = 0
    temp = 0
    threshold = 0
    while G <= 256:
        sc0 = P[0:G].sum()  # C0类像素所占面积的比例
        sc1 = P[G + 1:256].sum()  # C1类像素所占面积的比例
        u0 = (P[0:G] * range(0, G)).sum() / sc0  # C0类像素的平均灰度
        u1 = (P[G:256] * range(G, 256)).sum() / sc1  # C1类像素的平均灰度
        u = sc0 * u0 + sc1 * u1  # 整个图像的平均灰度
        v = sc0 * np.square(u - u0) + sc1 * np.square(u - u1)  # 方差
        if v > temp:
            temp = v
            threshold = G
        G += 1

    ostu_img = np.zeros(grayImg.shape, np.uint8)
    for i in range(grayImg.shape[0]):
        for j in range(grayImg.shape[1]):
            if grayImg[i, j] > threshold:
                ostu_img[i, j] = 255

    return ostu_img


def iteration(grayImg):
    N = grayImg.size
    P = np.array([0] * 256)
    for i in range(grayImg.shape[0]):
        for j in range(grayImg.shape[1]):
            P[grayImg[i, j]] += 1

    p = 0
    Ti = 0
    for i in range(0, 256):
        if p >= N / 2:
            Ti = i
            break
        p += P[i]
    P = P / N

    Tiand1 = Ti
    while 1:
        sc0 = P[0:Ti].sum()  # C0类像素所占面积的比例
        sc1 = P[Ti + 1:256].sum()  # C1类像素所占面积的比例
        u0 = (P[0:Ti] * range(0, Ti)).sum() / sc0  # C0类像素的平均灰度
        u1 = (P[Ti:256] * range(Ti, 256)).sum() / sc1  # C1类像素的平均灰度
        Tiand1 = 0.5 * (u0 + u1)
        if abs(Tiand1 - Ti) < 1:  # 设定值为1
            break
        Ti = int(Tiand1)
    iter_img = np.zeros(grayImg.shape, np.uint8)
    for i in range(grayImg.shape[0]):
        for j in range(grayImg.shape[1]):
            if grayImg[i, j] > Ti:
                iter_img[i, j] = 255

    return iter_img


inputs = cv2.imread('C:/Users/Jun/PycharmProjects/ostuAiteration/Image/2.jpg', cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('origin', 0)
cv2.resizeWindow('origin', 400, 700)
cv2.imshow("origin", inputs)
ostuImg = OSTU(inputs)
iterImg = iteration(inputs)
cv2.namedWindow('otsu', 0)
cv2.resizeWindow('otsu', 400, 700)
cv2.imshow("otsu", ostuImg)
cv2.namedWindow('iteration', 0)
cv2.resizeWindow('iteration', 400, 700)
cv2.imshow("iteration", iterImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
