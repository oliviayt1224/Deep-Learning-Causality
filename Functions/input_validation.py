def validation_coeff(coeff):
    try:
        float(coeff)

    except Exception:
        raise ValueError("The value of the coefficient should be a float. Please try again.")

    if float(coeff) <= 0 or float(coeff) >= 1:
        raise ValueError("The value of the coefficient should be in range of (0,1). Please try again.")

    return float(coeff)


def validation_T(T):
    try:
        float(T)

    except Exception:
        raise ValueError("The value of T should be a float. Please try again.")

    if float(T) <= 0:
        raise ValueError("The value of T should be larger than zero. Please try again.")

    return float(T)


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



