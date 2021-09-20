import cv2 as cv
import numpy as np

img = cv.imread('images/pixel_beauty.png')
cv.imshow('Basic', img)

def translate(src, x, y):
    trans = np.float32([[1, 0, x], [0, 1, y]])
    dim = (src.shape[1], src.shape[0])
    return cv.warpAffine(src, trans, dim)

translated = translate(img, 100, 100)
cv.imshow('Translate', translated)

def rotate(src, angle, rotPoint=None): 
    (height, width) = src.shape[:2]

    if rotPoint is None:
        rotPoint = (width // 2, height // 2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dim = (width, height)

    return cv.warpAffine(src, rotMat, dim)

rotated = rotate(img, 45)
cv.imshow('Rotated', rotated)
rrotated = rotate(rotated, 45)
cv.imshow('RRotated', rrotated)

flipOverX = cv.flip(img, 0)
cv.imshow('Flip over x', flipOverX)

flipOverY = cv.flip(img, 1)
cv.imshow('Flip over y', flipOverY)

flipBoth = cv.flip(img, -1)
cv.imshow('Flip over Both', flipBoth)

cv.waitKey(0)
