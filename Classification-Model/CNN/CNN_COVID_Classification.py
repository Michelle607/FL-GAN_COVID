# -*- coding: utf-8 -*-

"""
Created on Sun Dec 13 19:37:59 2020
This is CNN model used for COVID classification
@author: cdnguyen
"""

import os
import numpy as np
import cv2
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from collections import Counter
import efficientnet.tfkeras as efn
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import pandas as pd
from keras.preprocessing import image
from sklearn.model_selection import KFold
from keras.optimizers import gradient_descent_v2
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers

TrainImage = "train"
TestImage = "test"
Normalimages = os.listdir("E:/downloads/PycharmProjects_E/FL-GAN_COVID/CovidDataset/train/normal")
pneumonaimages = os.listdir("E:/downloads/PycharmProjects_E/FL-GAN_COVID/CovidDataset/train/pneumonia")
COVID19images = os.listdir("E:/downloads/PycharmProjects_E/FL-GAN_COVID/CovidDataset/train/covid")

print(len(Normalimages), len(pneumonaimages), len(COVID19images))
NUM_TRAINING_IMAGES = len(Normalimages) + len(pneumonaimages) + len(COVID19images)
print(NUM_TRAINING_IMAGES)

image_size = 32
BATCH_SIZE = 8
epochs = 10

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

data_path_train = "E:/downloads/PycharmProjects_E/FL-GAN_COVID/CovidDataset/train"
data_path_test = "E:/downloads/PycharmProjects_E/FL-GAN_COVID/CovidDataset/test"

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   zoom_range=0.2,
                                   rotation_range=15,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(data_path_train,
                                                 target_size=(image_size, image_size),
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical',
                                                 shuffle=True)

testing_set = test_datagen.flow_from_directory(data_path_test,
                                               target_size=(image_size, image_size),
                                               batch_size=BATCH_SIZE,
                                               class_mode='categorical',
                                               shuffle=False)


def define_model1():
    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     input_shape=(32, 32, 3),
                     activation='relu'))

    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(3, activation='softmax'))
    # compile model
    opt = gradient_descent_v2.SGD(lr=0.001, momentum=0.9)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def define_model():
    # Initializing the CNN
    classifier = Sequential()

    # Step 1 - Convolution
    # Extract features from the images
    classifier.add(Conv2D(32, (3, 3),
                          input_shape=(32, 32, 3),
                          activation='relu'))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dropout(0.5))

    classifier.add(Dense(3, activation="softmax"))

    # Compiling the CNN
    classifier.compile(optimizer='Adam',
                       loss='categorical_crossentropy',  # 'binary_crossentropy',
                       metrics=['accuracy'])
    return classifier


model = define_model1()

history = model.fit(training_set, validation_data=testing_set,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=epochs, verbose=1)

from sklearn.metrics import classification_report, confusion_matrix

# 전체 테스트 이미지 수 확인
num_test_images = testing_set.samples
steps = num_test_images // BATCH_SIZE + (num_test_images % BATCH_SIZE > 0)  # 나머지 데이터도 포함
Y_pred = model.predict(testing_set, steps=steps)
print("Y_pred shape:", Y_pred.shape)
predicted_classes = np.argmax(Y_pred, axis=1)
print("Predicted Classes Length:", len(predicted_classes))

true_classes = testing_set.classes
print("True classes length:", len(true_classes))
class_labels = list(testing_set.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')

confusion_matrix = confusion_matrix(testing_set.classes, y_pred)
print(confusion_matrix)

testing_set.class_indices.keys()
print(testing_set.class_indices.keys())
# To get better visual of the confusion matrix:
print('Classification Report')
