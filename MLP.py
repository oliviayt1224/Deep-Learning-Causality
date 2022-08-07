from tensorflow import keras
from tensorflow.keras import layers
from data import *


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def data_for_MLP(dataset, lag):

    dataset["Y_lagged"] = dataset["Y"].shift(periods=lag)
    dataset = dataset.dropna(axis=0, how='any')
    inseqX = np.array(dataset["X_lagged"])
    seqY = np.array(dataset["Y"])
    inseqY = np.array(dataset["Y_lagged"])

    inseqX = inseqX.reshape((len(inseqX), 1))
    inseqY = inseqY.reshape((len(inseqY), 1))

    if (dataset.shape[1] == 4):
        inseqZ = np.array(dataset["Z_lagged"])
        inseqZ = inseqZ.reshape((len(inseqZ), 1))
        X = np.hstack((inseqX, inseqY, inseqZ))

    if (dataset.shape[1] == 3):
        X = np.hstack((inseqX, inseqY))

    Y = seqY.reshape((len(seqY), 1))

    return X, Y


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
X, Y = data_for_MLP(dataset,5)
loss = MLP(X,Y)
print(loss)