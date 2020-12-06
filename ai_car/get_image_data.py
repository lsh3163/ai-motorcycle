__author__ = 'will'

import pickle
import numpy as np
import cv2
data = pickle.load( open( "trainingdata.p", "rb" ), encoding="latin1" )
n_images = len(data)
test, training = data[0:int(n_images/3)], data[int(n_images/3):]


f = open("./data.txt", 'r')
img_paths = []
labels = []
while True:
    line = f.readline()
    if not line: break
    print(line)
    img_path, label = line.split()
    img_paths.append(img_path)
    labels.append(label)
f.close()

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

def get_real_training_data():
    trX = []
    trY = []
    path = "./images/"
    for img_path, label in zip(img_paths, labels):
        img = cv2.imread(path+img_path)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC) / 255.0
        trX.append(img) 
        if label=="l":
            trY.append(0)
        elif label=="r":
            trY.append(2)
        elif label=="s":
            trY.append(1)
        
        flip_img = cv2.flip(img, 1)
        trX.append(flip_img)
        if label=="l":
            trY.append(2)
        elif label=="r":
            trY.append(0)
        elif label=="s":
            trY.append(1)

    trX = np.array(trX)
    return trX, np.array(trY)