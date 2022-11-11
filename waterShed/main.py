import numpy as np
import cv2
import matplotlib.pyplot as plt


def waterShed(image, gray, rgb):
    ret1, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dt = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret2, sure_fg = cv2.threshold(dt, 0.005 * dt.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    cv2.subtract(sure_bg, sure_fg)
    ret3, markers = cv2.connectedComponents(sure_fg)
    img = cv2.watershed(image, markers)
    return img


image = cv2.imread("C:/Users/Jun/PycharmProjects/waterShed/Image/2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.subplot(121)
plt.imshow(rgb)
plt.axis('off')
plt.subplot(122)
plt.imshow(waterShed(image, gray, rgb))
plt.axis('off')
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
