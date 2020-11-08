__author__ = 'will'

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

outputs = 1

from get_image_data import *

trX,trY = get_training_data()
teX,teY = get_test_data()

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

model=Sequential()
model.add(Dense(512, input_dim=np.shape(trX)[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trX, trY, epochs=50, batch_size=1)

Y_prediction = model.predict(teX).flatten()

for i in range(1000):
    label = teY[i]
    pred = Y_prediction[i]
    print("label:{:.2f}, pred:{:.2f}".format(label, pred))


def get_direction(img):
    print(img.shape)
#    img = np.array([np.reshape(img,img.shape**2)])
    ret =  model.predict(np.array([img]))
    return ret

# Predict direction with single image
dir=get_direction(teX[10])
print(dir[0][0])