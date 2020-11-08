import pickle
import numpy as np
from PIL import Image

data = pickle.load( open( "trainingdata.p", "rb" ), encoding="latin1")
n_images = len(data)

dataX = np.array([np.reshape(a[2],a[2].shape[0]**2) for a in data])
print(np.shape(dataX))

dataY = np.zeros((len(data)),dtype=np.float)
for i, data in enumerate(data):
    dataY[i] = float(data[0])
    
dataX = dataX.reshape((-1, 16, 16))
print(dataX)
print(dataY)

for i in range(0, 1000, 10):
    print(i)
    Image.fromarray(dataX[i]).save("img" + str(i) + "-" + str(dataY[i]) + ".jpeg")