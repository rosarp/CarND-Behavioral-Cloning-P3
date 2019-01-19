import cv2
import glob
import numpy as np
import pandas as pd
import os.path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from keras.models import Model
import utils
import pk_util

csv_list = glob.glob('../data/**/*.csv', recursive=True)
csv_rows = []
total_size = 0
for csv_file in csv_list:
    csv_file_data = pd.read_csv(csv_file)
    total_size = total_size + len(csv_file_data)
    for idx, row in csv_file_data.iterrows():
        columns = []
        image_path = '../data/' + csv_file.split('/')[-2] + '/IMG/' + row[0].split('/')[-1]
        columns.append(image_path)
        image_path = '../data/' + csv_file.split('/')[-2] + '/IMG/' + row[1].split('/')[-1]
        columns.append(image_path)
        image_path = '../data/' + csv_file.split('/')[-2] + '/IMG/' + row[2].split('/')[-1]
        columns.append(image_path)
        columns.append(row[3])
        columns.append(row[4])
        columns.append(row[5])
        columns.append(row[6])
        csv_rows.append(columns)


train_samples, validation_samples = train_test_split(csv_rows, test_size=0.2)
BATCH_SIZE = 32

# compile and train the model using the generator function
train_generator = utils.generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = utils.generator(validation_samples, batch_size=BATCH_SIZE)

model = utils.get_model()

history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator,  validation_steps=len(validation_samples), epochs=1, verbose=1)

model.save('model.h5')
