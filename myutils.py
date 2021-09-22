import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def translate(src, x, y):
    trans = np.float32([[1, 0, x], [0, 1, y]])
    dim = (src.shape[1], src.shape[0])
    return cv.warpAffine(src, trans, dim)


def rotate(src, angle, rotPoint=None):
    (height, width) = src.shape[:2]

    if rotPoint is None:
        rotPoint = (width // 2, height // 2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dim = (width, height)

    return cv.warpAffine(src, rotMat, dim)


def pltimshow(index, title, src, rows=2, cols=2, cmap='gray'):
    plt.subplot(rows, cols, index)
    plt.imshow(src, cmap)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


def pltshow():
    plt.show()
