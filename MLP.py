from tensorflow import keras
from tensorflow.keras import layers
from data_generation import *


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def data_for_MLP_Y(dataset):
    seqY = np.array(dataset["Y"])
    inseqY = np.array(dataset["Y_lagged"])
    X = np.hstack((inseqY))
    Y = seqY.reshape((len(seqY), 1))

    return X,Y

def data_for_MLP_XY(dataset):
    inseqX = np.array(dataset["X_lagged"])
    inseqY = np.array(dataset["Y_lagged"])

    inseqX = inseqX.reshape((len(inseqX), 1))
    inseqY = inseqY.reshape((len(inseqY), 1))

    seqY = np.array(dataset["Y"])

    X = np.hstack((inseqX, inseqY))
    Y = seqY.reshape((len(seqY), 1))

    return X,Y


def data_for_MLP_XYZ(dataset):

    inseqX = np.array(dataset["X_lagged"])
    inseqY = np.array(dataset["Y_lagged"])
    inseqZ = np.array(dataset["Z_lagged"])

    inseqX = inseqX.reshape((len(inseqX), 1))
    inseqY = inseqY.reshape((len(inseqY), 1))
    inseqZ = inseqZ.reshape((len(inseqZ), 1))

    seqY = np.array(dataset["Y"])

    X = np.hstack((inseqX, inseqY, inseqZ))
    Y = seqY.reshape((len(seqY), 1))

    return X,Y


def MLP(X,Y):
    dim = X.shape[1]
    model = keras.Sequential()
    model.add(layers.Dense(100, activation='relu', input_dim=dim))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')

    history = LossHistory()
    model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10,callbacks=[history])


    return history.losses[-1]

dataset = coupled_wiener_process(T=1, N=1000, alpha=0.5, lag=5, seed1=None, seed2=None)
print(dataset.shape[1])
X, Y = data_for_MLP_XY(dataset)
loss = MLP(X,Y)
print(loss)

# predict
# split out for both linear and non-linear, fix dataset