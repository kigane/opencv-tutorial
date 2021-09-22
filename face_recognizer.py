import cv2 as cv
import numpy as np
import os

people = os.listdir('faces/val')
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('faec_trained.yml')
corr_cnt = 0
for p in people:
    for i in range(5):
        # print(f'faces/val/{p}/{i+1}.jpg')
        img = cv.imread(f'faces/val/{p}/{i+1}.jpg')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        for (x,y,w,h) in face_rect:
            feature = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(feature)
            print(f'Label = {people[label]} with a confidence of {confidence}')

            cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
            cv.imshow('Recognized Face', img)

            if str(people[label]) == p:
                corr_cnt += 1

print(f'Correct rate = {corr_cnt/25}')


cv.waitKey(0)
