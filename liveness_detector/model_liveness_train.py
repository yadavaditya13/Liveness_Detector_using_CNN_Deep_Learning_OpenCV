import matplotlib

matplotlib.use(backend="Agg")

# importing packages

from my_script.livenessNet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True, help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")

args = vars(ap.parse_args())

# Initializing the learning rate, batch size and # of epoch to train

INIT_LR = 1e-4
BS = 8
EPOCHS = 50

# loading the image dataset and initializing the list of data and class images

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))

    # updating data and labels list respectively

    data.append(image)
    labels.append(label)

# convert my dataset into numpy array and preprosses by scaling pixel intensities to range [0, 1]
data = np.array(data, dtype="float") / 255.0

# Encoding the labels currently as strings, to integers and then one-hot encode them

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(y=labels, num_classes=2)

# partitioning the dataset for training and testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# construct the training image generator for data augmentation
# aug will be used further to generate images from our data
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# Initializing the optimizer and compiling model

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network

print("[INFO] training network for : {} epochs...".format(EPOCHS))
res = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY),
                          steps_per_epoch=len(trainX) // BS, epochs=EPOCHS)

# This ends the part of training model ... now let's display the results

print("[INFO] Evaluating Network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# Save the Network i.e Model to Disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# Save the Label Encoder to Disk as well
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

# Let's plot the training loss and accuracy

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), res.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), res.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), res.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), res.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])