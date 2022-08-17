def validation_coeff(coeff):
    if coeff <= 0 or coeff >= 1:
        raise ValueError("The value of the coefficient should be in range of (0,1). Please try again.")

    return coeff


def validation_T(T):
    if T <= 0:
        raise ValueError("The value of T should be larger than zero. Please try again.")

    return T


def validation_N(N):
    if N <= 0:
        raise ValueError("The value of N should be larger than zero. Please try again.")

    return N


def validation_lag(lag):
    if lag <= 0:
        raise ValueError("The value of lag should be larger than zero. Please try again.")

    return lag


def validation_N_lag(N, lag):
    if N <= lag:
        raise ValueError("The value of lag should be smaller than N. Please try again.")


def validation_num_exp(num_exp):
    if num_exp <= 0:
        raise ValueError("The number of experiments should be larger than zero. Please try again.")

    return num_exp


def validation_XY(initial_value):
    if initial_value <= 0 or initial_value >= 1:
        raise ValueError("The initial value should be in range of (0,1). Please try again.")

    return initial_value





