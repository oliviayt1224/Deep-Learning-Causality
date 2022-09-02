import numpy as np
import pandas as pd

# Generating a Wiener Process
def wiener_process(T, N, seed):

    if seed is not None:
        np.random.seed(seed)

    dt = T/N
    time = []
    wiener = []
    for i in range(N+1):
        time.append(i*dt)
        wiener.append(np.random.normal()*np.sqrt(i*dt))

    return np.array(wiener), np.array(time)


def coupled_wiener_process(T, N, alpha, lag, seed1=None, seed2=None):

    X, time = wiener_process(T=T, N=N, seed=seed1)
    V, time = wiener_process(T=T, N=N, seed=seed2)

    Y = []
    for i in range(lag, len(time)):
        Y.append(alpha*X[i-lag]+(1-alpha)*V[i])

    dataset = pd.DataFrame({"Y": Y, "X_lagged": X[:-lag], "X": X[lag:], "time": time[lag:]})
    dataset["Y_lagged"] = dataset["Y"].shift(periods=lag)
    dataset = dataset.dropna(axis=0, how='any')
    dataset.set_index("time", inplace=True)

    return dataset


def ternary_wiener_process(T, N, alpha, phi, beta, lag, seed1=None, seed2=None, seed3=None):
    Z, time = wiener_process(T=T, N=N, seed=seed1)
    V, time = wiener_process(T=T, N=N, seed=seed2)

    if seed3 is not None:
        np.random.seed(seed3)

    X = []
    Y = []

    for i in range(len(time)):
        X.append((1-phi)*np.random.normal()*np.sqrt(time[i])+phi*Z[i])

    for j in range(lag,len(time)):
        Y.append(alpha*X[j-lag]+beta*Z[j-lag]+(1-alpha-beta)*V[j])

    dataset = pd.DataFrame({"Y": Y, "X_lagged": X[:-lag], "X": X[lag:], "Z_lagged": Z[:-lag], "time": time[lag:]})
    dataset["Y_lagged"] = dataset["Y"].shift(periods=lag)
    dataset = dataset.dropna(axis=0, how='any')
    dataset.set_index("time", inplace=True)

    return dataset


def mapping_function(p, r):
    return r * p * (1 - p)


def coupling_function(p, epsilon, r):
    return (1 - epsilon) * mapping_function(p, r) + epsilon * mapping_function(mapping_function(p, r), r)


def coupled_logistic_map(X, Y, T, N, alpha, epsilon, r=4):
    # Generate index and initialise X, Y lists
    timesteps = list(np.linspace(0, T, N))
    (X, Y) = ([X], [Y])

    # Populate time series
    [X.append(mapping_function(X[n], r)) for n in range(N - 1)]

    [Y.append((1 - alpha) * mapping_function(Y[n], r) + alpha * coupling_function(X[n], epsilon, r)) for n in range(N - 1)]

    # Return DataFrame
    walk = pd.DataFrame({"Y": Y, "X": X, "time": timesteps})
    walk.set_index("time", inplace=True)

    return walk


def ternary_logistic_map(X, Y, T, N, alpha, epsilon, r=4):

    timesteps = list(np.linspace(0, T, N))
    (X, Y, Z) = ([X], [Y], [np.random.uniform()])

    # Populate time series
    [X.append(mapping_function(X[n], r)) for n in range(N - 1)]
    [Z.append(np.random.uniform()) for n in range(N-1)]
    [Y.append((1 - alpha) * mapping_function(Y[n], r) + alpha * coupling_function(X[n], epsilon, r)*Z[n]*Z[n]) for n in
     range(N - 1)]

    # Return DataFrame
    walk = pd.DataFrame({"Y": Y, "X": X, "Z": Z, "time": timesteps})
    walk.set_index("time", inplace=True)

    return walk


