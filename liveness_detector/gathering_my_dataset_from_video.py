import numpy as np
import argparse
import cv2
import os

# Inorder to have my own dataset i will be using two videofiles - real and - spoof and creating additional image
# datasets of my own

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", type=str, required=True, help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True, help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True, help="path to face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=16, help="# of frames to skip before applying face detection")

args = vars(ap.parse_args())

# loading our face detector model from disk

print("[INFO] loading face detector...")
prototxtPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

# initializing a pointer to the videofile

vs = cv2.VideoCapture(args["input"])
read = 0
# since i already had 160 images saved in both dataset/fake and dataset/real before i began writing the script
saved = 161

while True:
    (grabbed, frame) = vs.read()

    # if no frame found stop
    if not grabbed:
        break

    # number of frames read
    read += 1

    if read % args["skip"] != 0:
        continue

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    # ensure at least one face was found
    if len(detections) > 0:
        # get the index for that blob which has highest probability for being a face
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endy) = box.astype("int")
            face = frame[startY:endy, startX:endX]

            try:
                # writing the frame to disk
                _path = os.path.sep.join([args["output"], "{}.png".format(saved)])
                cv2.imwrite(_path, face)
                saved += 1
                print("[INFO] saved{} to disk...".format(_path))
            except cv2.error:
                continue


vs.release()
cv2.destroyAllWindows()