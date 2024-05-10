import cv2
import tensorflow as tf
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import pyttsx3
import os
import time
# Define class names based on your problem
class_names = [
    "Good Bye",
    "Good Luck",
    "Hello",
    "I Love You",
    "Losers",
    "Ok",
    "Silent",
    "Sorry",
    "Stop",
    "Victory"
]

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Open the camera
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("HandGesture.h5", "hello.txt")

offset = 20
imgSize = 300

# Parameters for tracking stable predictions
stable_frames_threshold = 10
max_pronunciations = 1
pronunciation_interval = 5
current_stable_frames = 0
prev_prediction = None
last_pronounced_prediction = None
pronunciation_count = 0
last_pronunciation_time = time.time() - pronunciation_interval

while True:
    ret, img = cap.read()

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    try:
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), cv2.FILLED)
            cv2.putText(imgOutput, class_names[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 255, 255), 3)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

            # Track stable predictions
            if prev_prediction is not None and prev_prediction == index:
                current_stable_frames += 1
            else:
                current_stable_frames = 0
                pronunciation_count = 0

            if current_stable_frames >= stable_frames_threshold and pronunciation_count < max_pronunciations:
                if time.time() - last_pronunciation_time >= pronunciation_interval:
                    if index != last_pronounced_prediction:
                        # Convert stable prediction to voice using pyttsx3
                        tts_text = class_names[index]
                        engine.say(tts_text)
                        engine.runAndWait()
                        last_pronunciation_time = time.time()
                        last_pronounced_prediction = index
                        pronunciation_count += 1

            prev_prediction = index

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
    except Exception as e:
        print("An error occurred:", str(e))

    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()