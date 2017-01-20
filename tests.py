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


#bin_range = np.arange(-1, 1.01, 0.05)
#print(np.histogram(dlog.steering, bin_range))

trainx_data = '../udacity-data/trainx.npy'
trainy_data = '../udacity-data/trainy.npy'
width = 320
height = 160
depth = 3
new_width = 64
new_height = 64

with open(trainx_data,'rb') as fx, open(trainy_data,'rb') as fy:
    x_train = np.load(fx)
    y_train = np.load(fy)

x_train = np.array([cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA) for img in x_train])
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2)

print(min(y_train), max(y_train))

print(x_train.shape)
print(y_train.shape)


def balance_data(x, y):
    random_transforms = [transforms.brightness, transforms.noise, transforms.shift]
    bins = np.arange(-1, 1.01, 0.1)
    angles_hist, _ = np.histogram(y, bins)
    max_bin = max(angles_hist)
    for steer_bin in range(len(angles_hist)):
        bin_num = angles_hist[steer_bin]
        if 0 < bin_num < max_bin:
            lower_bound = bins[steer_bin]
            upper_bound = bins[steer_bin + 1]
            bin_indexes = np.where((y >= lower_bound) & (y <= upper_bound))[0]
            x_bin, y_bin = [], []
            for i in range(max_bin-bin_num):
                bin_img_index = np.random.choice(bin_indexes)
                transform = np.random.choice(random_transforms)
                new_bin_x, new_bin_y = transform(x[bin_img_index], y[bin_img_index])
                x_bin.append(new_bin_x)
                y_bin.append(new_bin_y)
            x = np.concatenate((x, x_bin))
            y = np.concatenate((y, y_bin))
    return x, y

x_aug, y_aug = [], []
for img, angle in zip(x_train, y_train):
    img_blur, angle_blur = transforms.blur(img, angle)
    img_gray, angle_gray = transforms.gray(img, angle)
    img_mirror, angle_mirror = transforms.mirror(img, angle)
    x_aug.extend([img_blur, img_gray, img_mirror])
    y_aug.extend([angle_blur, angle_gray, angle_mirror])
x_train = np.concatenate((x_train, x_aug))
y_train = np.concatenate((y_train, y_aug))

x_train, y_train = balance_data(x_train, y_train)
'''

def rand_mirror(X, y):
    i = randint(0, len(X)-1)
    new_X = cv2.flip(X[i], 1)
    new_y = -y[i]
    return new_X, new_y

X_mirror, y_mirror = [], []
for i in range(15000):
    new_X, new_y = rand_mirror(x_train, y_train)
    X_mirror.append(new_X)
    y_mirror.append(new_y)
X_mirror, y_mirror = np.array(X_mirror), np.array(y_mirror)

def rand_noise(X, y):
    i = randint(0, len(X)-1)
    noise_max = 20
    noise_mask = np.random.randint(0, noise_max, (64, 64, 3), dtype='uint8')
    new_X = cv2.add(X[i], noise_mask)
    new_y = y[i]
    return new_X, new_y

X_noise, y_noise = [], []
for i in range(15000):
    new_X, new_y = rand_noise(x_train, y_train)
    X_noise.append(new_X)
    y_noise.append(new_y)
X_noise, y_noise = np.array(X_noise), np.array(y_noise)

x_train = np.concatenate((x_train, X_mirror, X_noise))
y_train = np.concatenate((y_train, y_mirror, y_noise))
'''
print(x_train.shape)
print(y_train.shape)

datagen = ImageDataGenerator(
    #rescale=1. / 255
    )

print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Convolution2D(8, 7, 7, input_shape=(new_height, new_width, depth)))
model.add(MaxPooling2D((2, 2)))
model.add(ELU())
model.add(BatchNormalization())
model.add(Convolution2D(8, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(ELU())
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
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
model.compile('adam', 'mean_squared_error', ['accuracy'])

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), samples_per_epoch=len(x_train),
                              nb_epoch=5, validation_data=datagen.flow(x_validation, y_validation, batch_size=32),
                              nb_val_samples=len(x_validation))

print(history)
val_preds = model.predict(x_validation)
print(min(val_preds), max(val_preds))

model.save_weights("model.h5", True)
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
