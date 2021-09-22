import cv2 as cv
import numpy as np
from myutils import pltimshow, pltshow

img = cv.imread('images/cat2.jpg')
resized = cv.resize(img, (800, 600), interpolation=cv.INTER_AREA)
cv.imshow('Cat', resized)

gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
pltimshow(1, 'Gray', gray)

laplacian = cv.Laplacian(gray, cv.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))
pltimshow(2, 'Laplacian', laplacian)

sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
pltimshow(3, 'Sobelx', sobelx)
pltimshow(4, 'Sobely', sobely)

pltshow()
