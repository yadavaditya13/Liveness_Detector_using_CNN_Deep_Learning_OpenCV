# This is the Script that will test out model on live cam

from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True, help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, required=True, help="path to face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

# Loading our face detector model from disk

print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNetFromCaffe(protoPath, weightsPath)

# Loading our Liveness detector model and label encoder from disk

print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())

# initializing the video stream and allowing the camera sensor to warmup

print("[INFO] We are going live...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# looping over the frames from live stream

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    # grabbing frame dimensions and converting it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # passing blob to face detector for detecting faces
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # looping over the detected faces in each frame
    for i in range(0, detections.shape[2]):

        # getting confidence for each detected faces in frame one by one
        confidence = detections[0, 0, i, 2]

        # filtering weak detections
        if confidence > args["confidence"]:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the detected bounding box does fall outside the
            # dimensions of the frame

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # extracting the face ROI's
            # and preprocessing out frame data

            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # passing ROI through trained liveness detector model

            preds = model.predict(face)[0]
            index = np.argmax(preds)
            label = le.classes_[index]

            # drawing label and bounding box on the frame

            label = "{}: {:.2f}%".format(label, preds[index] * 100)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # displaying the frame
    cv2.imshow("Frame: ", frame)
    key = cv2.waitKey(1) & 0xFF

    # if 'q' key is pressed end the stream
    if key == ord("q"):
        break

# Doing cleanUp Task
cv2.destroyAllWindows()
vs.stop()
