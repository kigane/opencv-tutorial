import cv2 as cv
import numpy as np
import myutils as u

img = cv.imread('images/wbb.png')
pltimg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
u.pltimshow(1, 'bingbing', pltimg, cmap=None)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
u.pltimshow(2, 'Gray bingbing', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
fact_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f'Numbers of faces found = {len(fact_rect)}')

for (x, y, w, h) in fact_rect:
    cv.rectangle(pltimg, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

u.pltimshow(3, 'Faces', pltimg)

u.pltshow()
