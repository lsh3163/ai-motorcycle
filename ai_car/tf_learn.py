# Copyright Reserved (2020).
# Donghee Lee, Univ. of Seoul
#
__author__ = 'will'

from keras.models import Sequential
from keras.layers import Dense
#from sklearn.model_selection import train_test_split

import numpy as np
#import pandas as pd
import tensorflow as tf
#import pickle
from get_image_data import *

class DNN_Driver():
    def __init__(self):
        self.trX = None
        self.trY = None
        self.teX = None
        self.teY = None
        self.model = None

    def tf_learn(self):
        self.trX, self.trY = get_training_data()
        self.teX, self.teY = get_test_data()
        seed = 0
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print(self.trX.shape, self.trY.shape)
        self.model=Sequential()
        self.model.add(Dense(512, input_dim=np.shape(self.trX)[1], activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation="tanh"))

        self.model.compile(loss='mean_squared_error', optimizer='adam')

        self.model.fit(self.trX, self.trY, epochs=1, batch_size=1)
        self.model.save_weights('./checkpoints/my_checkpoint')
        return

    def predict_direction(self, img):
        self.model.load_weights('./checkpoints/my_checkpoint')
        # self.model.save_weights('./checkpoints/my_checkpoint')

        
        # img = np.array([np.reshape(img,img.shape**2)])
        print(img.shape)
        ret =  self.model.predict(np.array([img]))
        print(ret, self.teY[10])
        return ret

    def get_test_img(self):
        img = self.teX[10]
        return img

dnn_driver = DNN_Driver()
dnn_driver.tf_learn()
img = dnn_driver.get_test_img()
dnn_driver.predict_direction(img)