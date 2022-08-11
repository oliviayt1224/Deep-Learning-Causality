from data_generation import *


# class LossHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = []
#
#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))

def data_for_MLP_Y(dataset):
    # seqY = np.array(dataset["Y"])
    # inseqY = np.array(dataset["Y_lagged"])
    X = np.array(dataset["Y_lagged"])
    X = X.reshape((len(X), 1))
    Y = np.array(dataset["Y"])

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

def training_testing_set(X,Y, percentage = 0.7):
    split_pos = round(X.shape[0] * percentage)
    if X.shape[1]>1:
        training_X = X[0:split_pos, :]
        testing_X = X[split_pos:, :]
    else:
        training_X = X[0:split_pos]
        testing_X = X[split_pos:]

    training_Y= Y[0:split_pos]
    testing_Y = Y[split_pos:]
    return training_X, testing_X, training_Y, testing_Y

# def MLP(X,Y):
#     training_X, testing_X, training_Y, testing_Y = training_testing_set(X, Y, percentage = 0.7)
#     dim = training_X.shape[1]
#     model = keras.Sequential()
#     model.add(layers.Dense(100, activation='relu', input_dim=dim))
#     model.add(layers.Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#
#     history = LossHistory()
#     model.fit(training_X, training_Y, validation_split=0.33, epochs=150, batch_size=10,callbacks=[history])
#     ypred = model.predict(testing_X)
#     resi = testing_Y - ypred
#     # return history.losses[-1]
#     return resi

def data_for_MLP_XY(dataset):
    X = np.array(dataset[["X_lagged","Y_lagged"]])
    # inseqY = np.array(dataset["Y_lagged"])
    # inseqX = inseqX.reshape((len(inseqX), 1))
    # inseqY = inseqY.reshape((len(inseqY), 1))

    Y = np.array(dataset["Y"])

    # X = np.hstack((inseqX, inseqY))
    # Y = seqY.reshape((len(seqY), 1))

    return X,Y

# dataset = coupled_wiener_process(T=1, N=100, alpha=0.5, lag=5, seed1=None, seed2=None)
# X, Y = data_for_MLP_XY(dataset)
# loss = MLP(X,Y)
# print(loss)


# predict
# split out for both linear and non-linear, fix dataset