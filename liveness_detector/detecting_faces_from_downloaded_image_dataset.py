import numpy as np
import argparse
from imutils import paths
import cv2
import os

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", type=str, required=True, help="path to input image directory")
ap.add_argument("-o", "--output", type=str, required=True, help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True, help="path to face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

# loading our face detector model from disk

print("[INFO] loading face detector...")
prototxtPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

# grab the list of images in our images directory, then initialize
# the list of data (i.e., images) and class images

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["input"]))
saved = 300

for imagePath in imagePaths:
    # lets get the images one by one
    image = cv2.imread(imagePath)

    # lets grab the image size
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

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
            face = image[startY:endy, startX:endX]

            # writing the frame to disk
            _path = os.path.sep.join([args["output"], "{}.png".format(saved)])
            cv2.imwrite(_path, face)
            saved += 1
            print("[INFO] saved{} to disk...".format(_path))
# The cleanUp
cv2.destroyAllWindows()