import pandas as pd
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D
from matplotlib.image import imread
import json
import matplotlib.pyplot as plt

train_data = '../udacity-data/train.p'
width = 320
height = 160
depth = 3

with open(train_data,'rb') as f_train_data:
    train = pickle.load(f_train_data)

X_train = train['features']
y_train = train['labels']

model = Sequential()
model.add(Flatten(input_shape=(height, width, depth)))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile('adam', 'mean_squared_error', ['accuracy'])

history = model.fit(X_train, y_train, batch_size=128, nb_epoch=10, validation_split=0.2)

print(history)

model.save_weights("model.h5", True)
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
