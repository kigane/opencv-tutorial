import cv2 as cv
import numpy as np

img = cv.imread('images/cat2.jpg')
resized = cv.resize(img, (800, 600), interpolation=cv.INTER_AREA)
cv.imshow('Cat', resized)

gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray, 125, 255, type=cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

for i in range(10):
    adp_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, i+1)
    cv.imshow(f'AdaptiveThreshold C={i+1}', adp_thresh)
    

cv.waitKey(0)
