import cv2 as cv
import numpy as np

img = cv.imread('images/pixel_beauty.png')
cv.imshow('Basic', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

b, g, r = cv.split(img)

cv.imshow('Blue', cv.merge([b, blank, blank]))
cv.imshow('Green', cv.merge([blank, g, blank]))
cv.imshow('Red', cv.merge([blank, blank, r]))

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged = cv.merge([b, g, r])
cv.imshow('Merged', merged)

cv.waitKey(0)
