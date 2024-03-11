#Copyright Anirban Kar (anirbankar21@gmail.com)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import cv2
import os
import numpy as np
from PIL import Image

path = os.path.dirname(os.path.abspath(__file__))

# Create the recognizer using the correct method
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(path + r'\trainer\trainer.yml')

cascadePath = path + r"\Classifiers\face.xml"  # Corrected path string
faceCascade = cv2.CascadeClassifier(cascadePath)

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX  # Corrected font usage

while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h, x:x+w])
        cv2.rectangle(im, (x-50, y-50), (x+w+50, y+h+50), (225, 0, 0), 2)

        if nbr_predicted == 1:
            nbr_predicted = 'Obama'
        elif nbr_predicted == 0:
            nbr_predicted = 'Jaron'

        cv2.putText(im, str(nbr_predicted)+"--"+str(conf), (x, y+h), font, 1.1, (0, 255, 0))  # Corrected function name
        cv2.imshow('im', im)
        cv2.waitKey(10)
