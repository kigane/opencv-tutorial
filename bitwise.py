import cv2 as cv
import numpy as np

img = cv.imread('images/cat.jpg')
resized = cv.resize(img, (400, 400), interpolation=cv.INTER_AREA)

blank = np.zeros((400, 400), dtype='uint8')

rect = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, thickness=-1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, thickness=-1)

cv.imshow('Rectangle', rect)
cv.imshow('Circle', circle)

weired = cv.bitwise_and(rect, circle)
cv.imshow('Bitwise and', weired)

weiredImg = cv.bitwise_and(resized, resized, mask=weired)
cv.imshow('Weired Img', weiredImg)

cv.waitKey(0)