#Maximum entropy
import numpy as np
import cv2

path = r"E:\Clion\test4.jpg"
imag = cv2.imdecode(np.fromfile(path),1)
imag = cv2.cvtColor(imag,cv2.COLOR_BGR2GRAY)

def calcGrayHist(image):
    rows, cols = image.shape
    grayHist = np.zeros([256])
    for r in range(rows):
        for c in range(cols):
            grayHist[int(image[r,c])] += 1

    return grayHist

def threshEntroy(image):
    rows, cols = image.shape
    grayHist = calcGrayHist(image)
    normgrayHist = grayHist / float(rows * cols)
    zeroCumuMoment = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0:
            zeroCumuMoment[k] = normgrayHist[k]
        else:
            zeroCumuMoment[k] = zeroCumuMoment[k - 1] + normgrayHist[k]
    entropy = np.zeros([256], np.float32)
    # calculate entropy
    for k in range(256):
        if k == 0:
            if normgrayHist[k] == 0:
                entropy[k] = 0
            else:
                entropy[k] = -normgrayHist[k] * np.log10(normgrayHist[k])
        else:
            if normgrayHist[k] == 0:
                entropy[k] = entropy[k - 1]
            else:
                entropy[k] = entropy[k - 1] - normgrayHist[k] * np.log10(normgrayHist[k])

    ft = np.zeros([256], np.float32)
    totalEntropy = entropy[255]
    for k in range(255):
        # 找最大值
        maxfornt = np.max(normgrayHist[:k + 1])
        maxback = np.max(normgrayHist[k + 1:256])
        if (maxfornt == 0 or zeroCumuMoment[k] == 0 or maxfornt == 1 or zeroCumuMoment[k] == 1 or totalEntropy == 0):
            ft1 = 0
        else:
            ft1 = entropy[k] / totalEntropy * (np.log10(zeroCumuMoment[k]) / np.log10(maxfornt))
        if (maxback == 0 or 1 - zeroCumuMoment[k] == 0 or maxback == 1 or 1 - zeroCumuMoment[k] == 1):
            ft2 = 0
        else:
            if totalEntropy == 0:
                ft2 = (np.log10(1 - zeroCumuMoment[k]) / np.log10(maxback))
            else:
                ft2 = (1 - entropy[k] / totalEntropy) * (np.log10(1 - zeroCumuMoment[k]) / np.log10(maxback))
        ft[k] = ft1 + ft2
    # find Maximum entropy
    thresloc = np.where(ft == np.max(ft))
    thresh = thresloc[0][0]
    return thresh

th= threshEntroy(np.array(imag))
_, th1 = cv2.threshold(imag, th, 255, cv2.THRESH_BINARY)
cv2.imshow("f", th1)
cv2.waitKey()

