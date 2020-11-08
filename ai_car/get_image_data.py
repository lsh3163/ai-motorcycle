__author__ = 'will'

import pickle
import numpy as np

data = pickle.load( open( "trainingdata.p", "rb" ), encoding="latin1" )
n_images = len(data)
test, training = data[0:int(n_images/3)], data[int(n_images/3):]

def get_training_data():

    trX = np.array([np.reshape(a[2],a[2].shape[0]**2) for a in training])
    print(np.shape(trX)[1])
    trY = np.zeros((len(training)),dtype=np.float)
    for i, data in enumerate(training):
        trY[i] = float(data[0])
    return trX, trY

def get_test_data():
    teX = np.array([np.reshape(a[2],a[2].shape[0]**2) for a in test])
    teY = np.zeros((len(test)),dtype=np.float)
    for i, data in enumerate(test):
        teY[i] = float(data[0])
    return teX,teY

