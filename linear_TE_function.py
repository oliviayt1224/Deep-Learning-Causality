import statsmodels.api as sm
from data import *

def linear_TE(X, Y, lag, df):
    # Initialise list to return TEs

    X_lagged = X + "_lag" + str(lag)
    Y_lagged = Y + "_lag" + str(lag)

    df[Y_lagged] = df[Y].shift(periods=lag)
    df[X_lagged] = df[X].shift(periods=lag)

    df = df.dropna(axis=0, how='any')

    joint_residuals = (sm.OLS(df[Y], sm.add_constant(df[[Y_lagged, X_lagged]])).fit().resid)
    independent_residuals = (sm.OLS(df[Y], sm.add_constant(df[Y_lagged])).fit().resid)

    # Use Geweke's formula for Granger Causality
    granger_causality = np.log(np.var(independent_residuals) / np.var(joint_residuals))

    # Calculate Linear Transfer Entropy from Granger Causality
    transfer_entropies = granger_causality / 2

    return transfer_entropies

def conditional_linear_TE(X, Y, Z, lag, df):
    # Initialise list to return TEs

    X_lagged = X + "_lag" + str(lag)
    Y_lagged = Y + "_lag" + str(lag)
    Z_lagged = Z + "_lag" + str(lag)

    df = df.dropna(axis=0, how='any')

    df[Y_lagged] = df[Y].shift(periods=lag)
    df[X_lagged] = df[X].shift(periods=lag)
    df[Z_lagged] = df[Z].shift(periods=lag)

    df = df.dropna(axis=0, how='any')

    joint_residuals = (sm.OLS(df[Y], sm.add_constant(df[[Y_lagged, X_lagged, Z_lagged]])).fit().resid)
    independent_residuals = (sm.OLS(df[Y], sm.add_constant(df[[Y_lagged, X_lagged]])).fit().resid)

    # Use Geweke's formula for Granger Causality
    granger_causality = np.log(np.var(independent_residuals) / np.var(joint_residuals))

    # Calculate Linear Transfer Entropy from Granger Causality
    transfer_entropies = granger_causality / 2

    return transfer_entropies