__author__ = 'will'

import pickle
import numpy as np
import cv2
data = pickle.load( open( "trainingdata.p", "rb" ), encoding="latin1" )
n_images = len(data)
test, training = data[0:int(n_images/3)], data[int(n_images/3):]

def get_training_data():

    trX = np.array([cv2.resize(np.reshape(a[2],a[2].shape[0]**2), (64, 64), interpolation=cv2.INTER_CUBIC) for a in training]) / 255.0
    print(np.shape(trX)[1])
    trY = np.zeros((len(training)),dtype=np.float)
    for i, data in enumerate(training):
        trY[i] = float(data[0]) + 1
    return trX, trY

def get_test_data():
    teX = np.array([cv2.resize(np.reshape(a[2],a[2].shape[0]**2), (64, 64), interpolation=cv2.INTER_CUBIC) for a in test]) / 255.0
    teY = np.zeros((len(test)),dtype=np.float)
    for i, data in enumerate(test):
        teY[i] = float(data[0]) + 1
    return teX,teY

