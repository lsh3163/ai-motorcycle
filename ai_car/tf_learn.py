# Copyright Reserved (2020).
# Donghee Lee, Univ. of Seoul
#
__author__ = 'will'
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
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
        print()
        # self.trX = self.trX.reshape((-1, 16, 16, 1))
        # self.teX = self.teX.reshape((-1, 16, 16, 1))
        print(self.trX.max(), self.teX.max())
        seed = 0
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print(self.trX.shape, self.trY.shape)
        self.model=Sequential()
        
        self.model.add(Conv2D(16, (3, 3), activation='tanh', input_shape=(16, 16, 1), padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, (3, 3), activation='tanh', padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, (3, 3), activation='tanh', padding="same"))
        self.model.add(Flatten())
        self.model.add(Dense(32, activation='tanh'))
        self.model.add(Dense(1, activation='tanh'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        print(self.model.summary())
        self.model.fit(self.trX, self.trY, epochs=20, batch_size=16)
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

    def get_score(self):
        self.model.load_weights('./checkpoints/my_checkpoint')
        return self.model.evaluate(self.teX, self.teY)
    def load_weights(self, path):
        self.model=Sequential()
        
        self.model.add(Conv2D(16, (3, 3), activation='tanh', input_shape=(16, 16, 1), padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, (3, 3), activation='tanh', padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, (3, 3), activation='tanh', padding="same"))
        self.model.add(Flatten())
        self.model.add(Dense(32, activation='tanh'))
        self.model.add(Dense(1, activation='tanh'))
        self.model.load_weights(path) 

if __name__ == '__main__': 
    dnn_driver = DNN_Driver()
    dnn_driver.tf_learn()
    img = dnn_driver.get_test_img()
    dnn_driver.predict_direction(img)
    print(dnn_driver.get_score())
