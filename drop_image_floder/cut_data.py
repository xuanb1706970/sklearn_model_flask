import re
import cv2
import numpy as np
import json
import time
import os

#Đường dẫn
path = "cut_face/name"

files = os.listdir("original_face/name/")
print(files)

count = 1

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
for i in range(len(files)):
    img = cv2.imread("original_face/name/"+files[i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        roi_color = img[y:y+h, x:x+w]
        cv2.imwrite(path + "/" + str(count) + ".jpg", roi_color)
        count = count + 1