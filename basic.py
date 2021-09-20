import cv2 as cv

img = cv.imread('images/pixel_beauty.png')
cv.imshow('Basic', img)

# 图片格式转换
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# 模糊处理
blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Edge Cascade 描边？
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny', canny)

# 扩展边缘
dilated = cv.dilate(canny, (7, 7), iterations=3)
cv.imshow('Dilate' ,dilated)

# 收缩边缘
eroded = cv.erode(dilated, (7, 7), iterations=3)
cv.imshow('Erode', eroded)

# 裁剪
cropped = img[100:400, 200:400]
cv.imshow('Crop', cropped)

cv.waitKey(0)
