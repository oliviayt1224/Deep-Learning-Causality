import numpy as np


def data_for_MLP_XY_Y(dataset, reverse=False):
    """ Process dataset into the required forms for MLP for two-variable case.

    Parameters
    ----------
    dataset : `pandas.DataFrame`
        input dataset.
    reverse :  `bool`, optional
        decide whether X or Y is the dependant variable.

    Returns
    -------
    X_jr : `pandas.DataFrame`
        data of two independent variables.
    Y_jr : `pandas.DataFrame`
        data of the dependent variable.
    X_ir : `pandas.DataFrame`
        data of one independent variable.
    Y_ir : `pandas.DataFrame`
        data of the dependent variable.
    """
    if reverse == False:
        data_ir = np.array(dataset[["Y_lagged", "Y"]])
        X_ir = data_ir[:, :-1]
        Y_ir = data_ir[:, -1:]

        data_jr = np.array(dataset[["X_lagged", "Y_lagged", "Y"]])
        X_jr = data_jr[:, :-1]
        Y_jr = data_jr[:, -1:]

    else:
        data_ir = np.array(dataset[["X_lagged", "X"]])
        X_ir = data_ir[:, :-1]
        Y_ir = data_ir[:, -1:]

        data_jr = np.array(dataset[["X_lagged", "Y_lagged", "X"]])
        X_jr = data_jr[:, :-1]
        Y_jr = data_jr[:, -1:]

    return X_jr, Y_jr, X_ir, Y_ir


def data_for_MLP_XYZ_XY(dataset, reverse=False):
    """ Process dataset into the required forms for MLP for three-variable case.

    Parameters
    ----------
    dataset : `pandas.DataFrame`
        input dataset.
    reverse :  `bool`, optional
        decide whether X or Y is the dependant variable.

    Returns
    -------
    X_jr : `pandas.DataFrame`
        data of three independent variables.
    Y_jr : `pandas.DataFrame`
        data of the dependent variable.
    X_ir : `pandas.DataFrame`
        data of two independent variables.
    Y_ir : `pandas.DataFrame`
        data of the dependent variable.
    """
    if reverse == False:
        data_jr = np.array(dataset[["X_lagged", "Y_lagged", "Z_lagged", "Y"]])
        X_jr = data_jr[:, :-1]
        Y_jr = data_jr[:, -1:]

        data_ir = np.array(dataset[["X_lagged", "Y_lagged", "Y"]])
        X_ir = data_ir[:, :-1]
        Y_ir = data_ir[:, -1:]

    else:
        data_jr = np.array(dataset[["X_lagged", "Y_lagged", "Z_lagged", "X"]])
        X_jr = data_jr[:, :-1]
        Y_jr = data_jr[:, -1:]

        data_ir = np.array(dataset[["X_lagged", "Y_lagged", "X"]])
        X_ir = data_ir[:, :-1]
        Y_ir = data_ir[:, -1:]

    return X_jr, Y_jr, X_ir, Y_ir


def training_testing_set_linear(dataset, percentage=0.7):
    """ Separate dataset into training and testing set for the linear method.

    Parameters
    ----------
    dataset : `pandas.DataFrame`
        input dataset.
    percentage : `float`, optional
        splitting percentage.
    Returns
    -------
    training_set : `pandas.DataFrame`
        output training set.
    testing_set : `pandas.DataFrame`
        output testing set.
    """
    split_pos = round(dataset.shape[0] * percentage)
    training_set = dataset.iloc[0:split_pos]
    testing_set = dataset.iloc[split_pos:]
    return training_set, testing_set


def training_testing_set_nonlinear(X, Y, percentage=0.7):
    """ Separate dataset into training and testing set for the nonlinear method.

    Parameters
    ----------
    X : `pandas.DataFrame`
        data of the independent variables.
    Y : `pandas.DataFrame`
        data of the dependent variable.
    percentage : `float`, optional
        splitting percentage.

    Returns
    -------
    training_X : `pandas.DataFrame`
        output training set for independent variables.
    testing_X : `pandas.DataFrame`
        output testing set for independent variables.
    training_Y : `pandas.DataFrame`
        output training set for dependent variable.
    testing_Y : `pandas.DataFrame`
        output testing set for dependent variable.
    """
    split_pos = round(X.shape[0] * percentage)

    if X.shape[1] > 1:
        training_X = X[0:split_pos, :]
        testing_X = X[split_pos:, :]
    else:
        training_X = X[0:split_pos]
        testing_X = X[split_pos:]

    training_Y = Y[0:split_pos]
    testing_Y = Y[split_pos:]
    return training_X, testing_X, training_Y, testing_Y
