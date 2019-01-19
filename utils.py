import cv2
import numpy as np
import sklearn
import os.path
from sklearn.utils import shuffle

import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Conv2D

def add_data(image, angle, images, angles):
    image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    angle = float(angle)
    images.append(image)
    angles.append(angle)
    # flipped data
    images.append(cv2.flip(image,1))
    angles.append(angle*-1.0)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                add_data(batch_sample[0], batch_sample[3], images, angles)
                correction = 0.2
                add_data(batch_sample[1], batch_sample[3] + correction, images, angles)
                add_data(batch_sample[2], batch_sample[3] - correction, images, angles)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def get_model():
    if os.path.isfile('model.h5'):
        # helps in retraining with new data
        model = load_model('model.h5')
    else:
        # first time run
        orgin_row, orgin_col, orgin_ch = 160, 320, 3  # Original image format
        row, col, ch = 90, 320, 3  # Trimmed image format

        model = Sequential()
        # Preprocess incoming data, centered around zero with small standard deviation 
        model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(orgin_row, orgin_col, orgin_ch)))
        # trim image to only see section with road
        model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(orgin_row, orgin_col, orgin_ch)))

        # Nvdia Architecture
        # Layer 1 - 24@5x5
        model.add(Conv2D(filters = 24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        model.add(Dropout(rate=0.5))

        # Layer 2 - 36@5x5
        model.add(Conv2D(filters = 36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        model.add(Dropout(rate=0.4))

        # Layer 3 - 48@5x5
        model.add(Conv2D(filters = 48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        model.add(Dropout(rate=0.3))

        # Layer 4 - 64@3x3
        model.add(Conv2D(filters = 64, kernel_size=(3, 3), activation='relu'))
        model.add(Dropout(rate=0.2))

        # Layer 5 - 64@3x3
        model.add(Conv2D(filters = 64, kernel_size=(3, 3), activation='relu'))
        model.add(Dropout(rate=0.2))

        model.add(Flatten())
        # Fully connected layers
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(rate=0.3))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))

        model.compile(loss='mse', optimizer='adam')
    return model
