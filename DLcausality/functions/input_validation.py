def validation_coeff(coeff):
    try:
        float(coeff)
    except Exception:
        raise ValueError("The value of the coefficient should be a number in range of (0,1). Please try again.")

    if coeff <= 0 or coeff >= 1:
        raise ValueError("The value of the coefficient should be a number in range of (0,1). Please try again.")

    return coeff


def validation_T(T):
    try:
        float(T)
    except Exception:
        raise ValueError("The value of T should be a number larger than zero. Please try again.")

    if T <= 0:
        raise ValueError("The value of T should be a number larger than zero. Please try again.")

    return T


def validation_N(N):
    try:
        int(N)
    except Exception:
        raise ValueError("The value of N should be an integer larger than zero. Please try again.")

    if N <= 0:
        raise ValueError("The value of N should be an integer larger than zero. Please try again.")

    return N


def validation_lag(lag):
    try:
        int(lag)
    except Exception:
        raise ValueError("The value of lag should be an integer larger than zero. Please try again.")

    if lag <= 0:
        raise ValueError("The value of lag should be an integer larger than zero. Please try again.")

    return lag


def validation_N_lag(N, lag):
    if N <= lag:
        raise ValueError("The value of lag should be smaller than N. Please try again.")


def validation_num_exp(num_exp):
    try:
        int(num_exp)
    except Exception:
        raise ValueError("The number of experiments should be an integer larger than zero. Please try again.")

    if num_exp <= 0:
        raise ValueError("The number of experiments should be an integer larger than zero. Please try again.")

    return num_exp


def validation_XY(initial_value):
    try:
        float(initial_value)
    except Exception:
        raise ValueError("The initial value should be a float in range of (0,1). Please try again.")

    if initial_value <= 0 or initial_value >= 1:
        raise ValueError("The initial value should be a float in range of (0,1). Please try again.")

    return initial_value





