import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data import *

lag = 5
dataset = coupled_wiener_process(T = 1, N = 1000, alpha = 0.5, lag = lag, seed1=None, seed2=None)
dataset["Y_lagged"] = dataset["Y"].shift(periods=lag)
dataset = dataset.dropna(axis=0, how='any')
inseqX = np.array(dataset["X_lagged"])
seqY = np.array(dataset["Y"])
inseqY = np.array(dataset["Y_lagged"])

inseqX = inseqX.reshape((len(inseqX), 1))
inseqY = inseqY.reshape((len(inseqY), 1))

X = np.hstack((inseqX, inseqY))
Y = seqY.reshape((len(seqY), 1))

model = keras.Sequential()
model.add(layers.Dense(100, activation='relu', input_dim=2))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)