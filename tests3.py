import pandas as pd
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from matplotlib.image import imread
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
from random import randint
import transforms
tf.python.control_flow_ops = tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import transforms

trainx_data = '../udacity-data/vgg16features.npy'
trainy_data = '../udacity-data/vgg16labels.npy'

with open(trainx_data,'rb') as fx, open(trainy_data,'rb') as fy:
    x_train = np.load(fx)
    y_train = np.load(fy)

datagen = ImageDataGenerator(
    #rescale=1. / 255
    )

print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(ELU())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(ELU())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(ELU())
model.add(Dense(1))
model.add(Activation('tanh'))
model.compile('adam', 'mean_squared_error', ['accuracy'])

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), samples_per_epoch=len(x_train),
                              nb_epoch=5)

print(history)

model.save_weights("model.h5", True)
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
