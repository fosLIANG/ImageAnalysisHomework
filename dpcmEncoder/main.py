import cv2
import numpy as np
import matplotlib.pyplot as plt
import io

img = cv2.imread('C:/Users/13340/Desktop/dpcmEncoder/Image/1.jpg')
h, w = img.shape[:2]  # 高，宽
y12 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
gt, g, k = cv2.split(y12)
k = cv2.resize(k, (k.shape[1] // 2, k.shape[0] // 2))  # 要注意 w或h 为单数的情况
g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2))
f = io.BytesIO()
f.write(gt.tobytes())
f.write(k.tobytes())
f.write(g.tobytes())
f.seek(0)
img_np = np.frombuffer(f.read(), np.uint8)
lenimg = img_np.size
lenimg1 = h * w
gt = img_np[:lenimg1]
k = img_np[lenimg1:(lenimg - lenimg1) // 2 + lenimg1]
g = img_np[(lenimg - lenimg1) // 2 + lenimg1:]
img3 = np.zeros(lenimg1, np.uint16)
kildech = np.zeros(lenimg1, np.uint16)


def quantization(quantizer, gt, kildech, img3, k, g):
    dr = 512 / (1 << quantizer)
    for i in range(h):
        for j in range(w):
            if j == 0:
                kil = gt[j + i * w] - 128  # 计算预测误差
                kildech[j + i * w] = (kil + 255) / dr  # 量化预测误差
                img3[j + i * w] = (kildech[j + i * w] - 255 / dr) * dr + 128
                if img3[j + i * w] > 255:
                    img3[j + i * w] = 255  # 防止重建像素超过255
                kildech[j + i * w] = kildech[j + i * w] * dr / 2

            else:
                kil = gt[j + i * w] - img3[j + i * w - 1]  # 计算预测误差
                kildech[j + i * w] = (kil + 255) / dr  # 量化
                img3[j + i * w] = (kildech[j + i * w] - 255 / dr) * dr + img3[j + i * w - 1]
                kildech[j + i * w] = kildech[j + i * w] * dr / 2
                if img3[j + i * w] > 255:
                    img3[j + i * w] = 255

    img3 = img3.astype(np.uint8)
    kildech = kildech.astype(np.uint8)
    gt = gt.reshape((h, w))
    kildech = kildech.reshape((h, w))
    img3 = img3.reshape((h, w))
    k = k.reshape((h // 2, w // 2))
    g = g.reshape((h // 2, w // 2))
    g4 = cv2.resize(k, (w, h))
    g5 = cv2.resize(g, (w, h))
    y12 = cv2.merge((gt, g5, g4))
    agt = cv2.cvtColor(y12, cv2.COLOR_YCrCb2BGR)
    fdf = cv2.merge((kildech, g5, g4))
    agtpr = cv2.cvtColor(fdf, cv2.COLOR_YCrCb2BGR)
    fdfe = cv2.merge((img3, g5, g4))
    agtr = cv2.cvtColor(fdfe, cv2.COLOR_YCrCb2BGR)
    return agtr, agtpr, agt

res = quantization(1, gt, kildech, img3, k, g)
agtr = res[0]
agtpr = res[1]
agt = res[2]
plt.subplot(241), plt.imshow(cv2.cvtColor(agtpr, cv2.COLOR_BGR2RGB), cmap='gray'),
plt.title('D_1bit'), plt.axis('off')
plt.subplot(245), plt.imshow(cv2.cvtColor(agtr, cv2.COLOR_BGR2RGB), cmap='gray'),
plt.title('Re_1bit'), plt.axis('off')
res = quantization(2, gt, kildech, img3, k, g)
agtr = res[0]
agtpr = res[1]
agt = res[2]
plt.subplot(242), plt.imshow(cv2.cvtColor(agtpr, cv2.COLOR_BGR2RGB), cmap='gray'),
plt.title('D_2bit'), plt.axis('off')
plt.subplot(246), plt.imshow(cv2.cvtColor(agtr, cv2.COLOR_BGR2RGB), cmap='gray'),
plt.title('Re_2bit'), plt.axis('off')
res = quantization(4, gt, kildech, img3, k, g)
agtr = res[0]
agtpr = res[1]
agt = res[2]
plt.subplot(243), plt.imshow(cv2.cvtColor(agtpr, cv2.COLOR_BGR2RGB), cmap='gray'),
plt.title('D_4bit'), plt.axis('off')
plt.subplot(247), plt.imshow(cv2.cvtColor(agtr, cv2.COLOR_BGR2RGB), cmap='gray'),
plt.title('Re_4bit'), plt.axis('off')
res = quantization(8, gt, kildech, img3, k, g)
agtr = res[0]
agtpr = res[1]
agt = res[2]
plt.subplot(244), plt.imshow(cv2.cvtColor(agtpr, cv2.COLOR_BGR2RGB), cmap='gray'),
plt.title('D_8bit'), plt.axis('off')
plt.subplot(248), plt.imshow(cv2.cvtColor(agtr, cv2.COLOR_BGR2RGB), cmap='gray'),
plt.title('Re_8bit'), plt.axis('off')
plt.show()
