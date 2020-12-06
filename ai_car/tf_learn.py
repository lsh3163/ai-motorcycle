# Copyright Reserved (2020).
# Donghee Lee, Univ. of Seoul
#
__author__ = 'will'
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Conv2D, Flatten, MaxPooling2D, Dropout
#from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
#import pandas as pd
import tensorflow as tf
#import pickle
from get_image_data import *
 
 
 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



class DNN_Driver():
    def __init__(self):
        self.trX = None
        self.trY = None
        self.teX = None
        self.teY = None
        self.model = None
        self.model=Sequential()
        
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3), padding="same"))
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.8))
        self.model.add(Conv2D(64, (3, 3), activation='tanh', padding="same"))
        self.model.add(Conv2D(64, (3, 3), activation='tanh', padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.8))

        self.model.add(Conv2D(128, (3, 3), activation='tanh', padding="same"))
        self.model.add(Conv2D(128, (3, 3), activation='tanh', padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.8))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.8))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def tf_learn(self):
        # self.trX, self.trY = get_training_data()
        self.trX, self.trY = get_real_training_data()
        # self.teX, self.teY = get_test_data()
        self.trX = self.trX.reshape((-1, 64, 64, 3))
        # self.teX = self.teX.reshape((-1, 64, 64, 1))
        print(self.trX.max())
        seed = 0
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print(self.trX.shape, self.trY.shape)
        
        
        print(self.model.summary())
        early_stopping_monitor = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=5,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True
        )
        self.model.fit(self.trX, self.trY, epochs=30, batch_size=16, validation_split=0.1, callbacks=[early_stopping_monitor])
        self.model.save_weights('./checkpoints/my_checkpoint')
        return

    def predict_direction(self, img):
        self.model.load_weights('./checkpoints/my_checkpoint')
        
        print(img.shape)
        ret =  self.model.predict(np.array([img]))
        return ret

    def get_test_img(self):
        img = self.trX[10]
        return img

    def get_score(self):
        self.model.load_weights('./checkpoints/my_checkpoint')
        return self.model.evaluate(self.teX, self.teY)
    def load_weights(self, path):
        
        self.model.load_weights(path) 

if __name__ == '__main__': 
    dnn_driver = DNN_Driver()
    dnn_driver.tf_learn()
    img = dnn_driver.get_test_img()
    print(dnn_driver.predict_direction(img))
    # print(dnn_driver.get_score())
