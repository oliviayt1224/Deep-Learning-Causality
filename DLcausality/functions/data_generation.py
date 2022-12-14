import numpy as np
import pandas as pd
from DLcausality.functions.input_validation import *


def wiener_process(T, N, seed=None):
    """ Generate a wiener process.

    Parameters
    ----------
    T : `float`
        time length of the generated series.
    N : `int`
        time step of the generated series.
    seed : `int`, optional
        the number used to initialize the random number generator.

    Returns
    -------
    wiener : `numpy.ndarray`
        the series of a wiener process.
    time : `numpy.ndarray`
        the series of time.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T/N
    time = []
    wiener = []
    for i in range(N+1):
        time.append(i*dt)
        wiener.append(np.random.normal()*np.sqrt(i*dt))

    wiener = np.array(wiener)
    time = np.array(time)

    return wiener, time


def coupled_wiener_process(T, N, alpha, lag, seed1=None, seed2=None):
    """ Generate a coupled wiener process.

    Parameters
    ----------
    T : `float`
        time length of the generated series.
    N : `int`
        time step of the generated series.
    alpha : `float`
        coefficient with a range of [0,1]
    lag : `int`
        time lag.
    seed1 : `int`, optional
        the number used to initialize the random number generator.
    seed2 : `int`, optional
        the number used to initialize the random number generator.

    Returns
    -------
    dataset : `pandas.DataFrame`
        coupled wiener process.
    """
    validation_T(T)
    validation_coeff(alpha)
    validation_N(N)
    validation_lag(lag)
    validation_N_lag(N, lag)

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
    """ Generate a ternary wiener process.

    Parameters
    ----------
    T : `float`
        time length of the generated series.
    N : `int`
        time step of the generated series.
    alpha : `float`
        coefficient with a range of [0,1]
    phi : `float`
        coefficient with a range of [0,1]
    beta : `float`
        coefficient with a range of [0,1]
    lag : `int`
        time lag.
    seed1 : `int`, optional
        the number used to initialize the random number generator.
    seed2 : `int`, optional
        the number used to initialize the random number generator.
    seed3 : `int`, optional
        the number used to initialize the random number generator.

    Returns
    -------
    dataset : `pandas.DataFrame`
        ternary wiener process.
    """
    validation_T(T)
    validation_coeff(alpha)
    validation_coeff(phi)
    validation_coeff(beta)
    validation_N(N)
    validation_lag(lag)
    validation_N_lag(N, lag)

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


def mapping_function(p, r=4):
    """ Mapping function f for logistic maps.

    Parameters
    ----------
    p : `float`
        previous value.
    r : `float`, optional
        positive influencing factor.

    Returns
    -------
    new_value : `float`
        generated new value.
    """
    new_value = r * p * (1 - p)
    return new_value


def coupling_function(p, epsilon, r=4):
    """ Coupling function g for logistic maps.

    Parameters
    ----------
    p : `float`
        previous value.
    epsilon : `float`
        coefficient with a range of [0,1]
    r : `float`, optional
        positive influencing factor.

    Returns
    -------
    new_value : `float`
        generated new value.
    """
    return (1 - epsilon) * mapping_function(p, r) + epsilon * mapping_function(mapping_function(p, r), r)


def coupled_logistic_map(X, Y, T, N, alpha, epsilon, r=4):
    """ Generate coupled logistic maps.

    Parameters
    ----------
    X : `float`
        initial value of X.
    Y : `float`
        initial value of Y.
    T : `float`
        time length of the generated series.
    N : `int`
        time step of the generated series.
    alpha : `float`
        coefficient with a range of [0,1]
    epsilon : `float`
        coefficient with a range of [0,1]
    r : `float`, optional
        positive influencing factor.

    Returns
    -------
    walk : `pandas.DataFrame`
        coupled logistic maps.
    """
    validation_XY(X)
    validation_XY(Y)
    validation_T(T)
    validation_coeff(alpha)
    validation_coeff(epsilon)
    validation_N(N)
    validation_N_lag(N, 1)

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
    """ Generate ternary logistic maps.

    Parameters
    ----------
    X : `float`
        initial value of X.
    Y : `float`
        initial value of Y.
    T : `float`
        time length of the generated series.
    N : `int`
        time step of the generated series.
    alpha : `float`
        coefficient with a range of [0,1]
    epsilon : `float`
        coefficient with a range of [0,1]
    r : `float`, optional
        positive influencing factor.

    Returns
    -------
    walk : `pandas.DataFrame`
        ternary logistic maps.
    """
    validation_XY(X)
    validation_XY(Y)
    validation_T(T)
    validation_coeff(alpha)
    validation_coeff(epsilon)
    validation_N(N)
    validation_N_lag(N, 1)

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


