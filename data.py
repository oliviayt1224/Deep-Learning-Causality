import numpy as np
import pandas as pd
import copy


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
    for i in range(lag,len(time)):
        Y.append(alpha*X[i-lag]+(1-alpha)*V[i])

    dataset = pd.DataFrame({"Y": Y, "X": X[lag:], "time": time[lag:]})
    dataset.set_index("time", inplace=True)

    return dataset


def tenary_wiener_process(T, N, alpha, phi, beta, lag, seed1=None, seed2=None, seed3=None):
    Z, time = wiener_process(T=T, N=N, seed=seed1)
    V, time = wiener_process(T=T, N=N, seed=seed2)

    if seed3 is not None:
        np.random.seed(seed3)

    X = []
    Y = []

    for i in range(len(time)):
        X.append(phi*np.random.normal()*np.sqrt(time[i])+(1-phi)*Z[i])

    for j in range(lag,len(time)):
        Y.append(alpha*X[j-lag]+beta*Z[j-lag]+(1-alpha-beta)*V[j])

    dataset = pd.DataFrame({"Y": Y, "X": X[lag:], "Z":Z[lag:], "time": time[lag:]})
    dataset.set_index("time", inplace=True)

    return dataset


def mapping_function(p, r):
    return r * p * (1 - p)


def coupling_function(p, epsilon, r):
    return (1 - epsilon) * mapping_function(p, r) + epsilon * mapping_function(mapping_function(p, r), r)


def coupled_logistic_map(S1, S2, T, N, alpha, epsilon, r=4):
    # Generate index and initialise S1, S2 lists
    timesteps = list(np.linspace(0, T, N))
    (S1, S2) = ([S1], [S2])

    # Populate time series
    [S1.append(mapping_function(S1[n], r)) for n in range(N - 1)]

    [S2.append((1 - alpha) * mapping_function(S2[n], r) + alpha * coupling_function(S1[n], epsilon, r)) for n in range(N - 1)]

    # Return DataFrame
    walk = pd.DataFrame({"S2": S2, "S1": S1, "time": timesteps})
    walk.set_index("time", inplace=True)

    return walk
