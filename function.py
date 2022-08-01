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


# data1 = coupled_wiener_process(T=1, N=100, alpha=0.5, lag=5)
# col_names = data1.columns.values.tolist()
# print(linear_TE(col_names[1], col_names[0], 5, data1))
#
data2 = tenary_wiener_process(T=1, N=100, alpha=0.2, phi=0.5, beta=0.5, lag=5)
col_names = data2.columns.values.tolist()
print("conditional_linear_TE:")
print(conditional_linear_TE(col_names[1], col_names[0], col_names[2], 5, data2))
print("normal_linear_TE:")
print(linear_TE(col_names[1], col_names[0], 5, data2))

# data3 = coupled_logistic_map(S1 = 0.5, S2 = 0.5, T = 1, N = 100, alpha = 0.5, epsilon = 0.5, r=4)
# col_names = data3.columns.values.tolist()
# print(linear_TE(col_names[1], col_names[0], 5, data3))