import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from matplotlib.image import imread
import json
import matplotlib.pyplot as plt

dire = '../udacity-data/'
log = pd.read_csv(dire + 'driving_log.csv')
width = 320
height = 160
depth = 3
steer_right = log.iloc[55]
steer_left = log.iloc[1181]
steer_straight = log.iloc[0]

img_right = imread(dire + steer_right.center)
img_left = imread(dire + steer_left.center)
img_straight = imread(dire + steer_straight.center)

X_train = np.array([img_right, img_left, img_straight])
y_train = np.array([steer_right.steering, steer_left.steering, steer_straight.steering])


model = Sequential()
model.add(Flatten(input_shape=(height, width, depth)))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile('sgd', 'mean_squared_error', ['accuracy'])
history = model.fit(X_train, y_train, batch_size=128, nb_epoch=10, validation_split=0.0)
print(history)
print(y_train)
print(model.predict(X_train))
model.save_weights("model.h5", True)
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
