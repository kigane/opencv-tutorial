import os
import cv2 as cv
import numpy as np

people = os.listdir('faces/train')
DIR = 'faces/train'
features = []
labels =[]

haar_cascade = cv.CascadeClassifier('haar_face.xml')

def create_train():
    for p in people:
        path = os.path.join(DIR, p)
        label = people.index(p)
        
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            raw = cv.imread(img_path)
            gray = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
            fact_rect = haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in fact_rect:
                features.append(gray[y:y+h, x:x+w])
                labels.append(label)

create_train()

features = np.array(features, dtype='object')
labels = np.array(labels)

# instantiate
face_recognizer = cv.face.LBPHFaceRecognizer_create()
# train
face_recognizer.train(features, labels)

face_recognizer.save('faec_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

print('------------ Trainning done ------------')