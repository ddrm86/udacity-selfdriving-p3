import pandas as pd
import numpy as np
from matplotlib.image import imread
import pickle

base_dir = '../my-data/'
driving_log_file = 'driving_log.csv'
dlog = pd.read_csv(base_dir + driving_log_file)

camera_offset = 0.25
X_train = []
y_train = []
for index, row in dlog.iterrows():
    steering = row.steering
    center_img, center_steering = imread(base_dir + row.center), steering
    left_img, left_steering = imread(base_dir + row.left), steering + camera_offset
    right_img, right_steering = imread(base_dir + row.right), steering - camera_offset
    X_train.extend([center_img, left_img, right_img])
    y_train.extend([center_steering, left_steering, right_steering])
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape, y_train.shape)
dic = dict()
dic['features'] = X_train
dic['labels'] = y_train
with open(base_dir + 'train.p', 'wb') as f:
    pickle.dump(dic, f)
