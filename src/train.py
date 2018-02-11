from autoencoder import *
import numpy as np

n_hidden = 300
feat_size = 13
learning_rate = 1e-3
frames = 1500
batch_size = 40
D = 100

sampleX = np.random.uniform(size=(batch_size*100, frames, feat_size))
sampleLen = np.full((batch_size*100,), frames)

autoencoder = Autoencoder(n_hidden, feat_size, learning_rate, frames, batch_size)

autoencoder.initPlaceholders()
autoencoder.initParams()
autoencoder.model()
autoencoder.calculateCost()

autoencoder.training(10, sampleX, sampleLen)  
