import cv2
import numpy as np

path = r"E:\Clion\test.jpg"
image = cv2.imdecode(np.fromfile(path),1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def Otsu2Threshold(src):
    Threshold1 = 0
    Threshold2 = 0
    weight, height = np.shape(src)
    hest = np.zeros([256], dtype=np.int32)
    for row in range(weight):
        for col in range(height):
            pv = src[row, col]
            hest[pv] += 1
    tempg = -1
    N_blackground = 0
    N_object = 0
    N_all = weight * height
    for i in range(256):
        N_object += hest[i]
        for k in range(i, 256, 1):
            N_blackground += hest[k]
        for j in range(i, 256, 1):
            gSum_object = 0
            gSum_middle = 0
            gSum_blackground = 0

            N_middle = N_all - N_object - N_blackground
            w0 = N_object / N_all
            w2 = N_blackground / N_all
            w1 = 1 - w0 - w2
            for k in range(i):
                gSum_object += k * hest[k]
            u0 = gSum_object / N_object
            for k in range(i + 1, j, 1):
                gSum_middle += k * hest[k]
            u1 = gSum_middle / (N_middle + 0.000001)

            for k in range(j + 1, 256, 1):
                gSum_blackground += k * hest[k]
            u2 = gSum_blackground / (N_blackground + 0.000001)

            u = w0 * u0 + w1 * u1 + w2 * u2
            g = w0 * (u - u0) * (u - u0) + w1 * (u - u1) * (u - u1) + w2 * (u - u2) * (u - u2)
            if tempg < g:
                tempg = g
                Threshold1 = i
                Threshold2 = j
            N_blackground -= hest[j]
    return Threshold1, Threshold2

_ , th2 = Otsu2Threshold(gray)
th2 = max(th2, 20)     #Lower bound = 20
ret1, th1 = cv2.threshold(gray, th2, 255, cv2.THRESH_BINARY)
cv2.imshow("thr", th1)
cv2.waitKey()