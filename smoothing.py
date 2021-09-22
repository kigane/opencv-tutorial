import cv2 as cv
import numpy as np

def resizeFrame(src, scale=0.75):
    dim = (int(src.shape[1] * scale), int(src.shape[0] * scale))
    return cv.resize(src, dim, interpolation=cv.INTER_AREA)

raw = cv.imread('images/cat.jpg')
img = resizeFrame(raw, scale=0.5)
cv.imshow('Cat', img)

average = cv.blur(img, (3, 3))
cv.imshow('Average', average)

gaussian = cv.GaussianBlur(img, (3, 3), 1)
cv.imshow('GassianBlur', gaussian)

median = cv.medianBlur(img, 3)
cv.imshow('Median', median)

bilateral = cv.bilateralFilter(img, 9, 75, 75)
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)
