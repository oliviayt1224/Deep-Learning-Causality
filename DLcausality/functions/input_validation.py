def validation_coeff(coeff):
    """ Validate coefficient.

    Parameters
    ----------
    coeff : `float`
        coefficient needed to be validated.

    Returns
    -------
    coeff : `float`
        output right coefficient.

    Raises
    ------
    ValueError
        if the coefficient is not a number or out of range.
    """
    try:
        float(coeff)
    except Exception:
        raise ValueError("The value of the coefficient should be a number in range of [0,1]. Please try again.")

    if coeff < 0 or coeff > 1:
        raise ValueError("The value of the coefficient should be a number in range of [0,1]. Please try again.")

    return coeff


def validation_T(T):
    """ Validate T.

    Parameters
    ----------
    T : `float`
        time length of the generated series.

    Returns
    -------
    T : `float`
        output right T.

    Raises
    ------
    ValueError
        if T is not a number or out of range.
    """
    try:
        float(T)
    except Exception:
        raise ValueError("The value of T should be a number larger than zero. Please try again.")

    if T <= 0:
        raise ValueError("The value of T should be a number larger than zero. Please try again.")

    return T


def validation_N(N):
    """ Validate N.

    Parameters
    ----------
    N : `int`
        time step of the generated series.

    Returns
    -------
    N : `int`
        output right N.

    Raises
    ------
    ValueError
        if N is not an integer or out of range.
    """
    try:
        int(N)
    except Exception:
        raise ValueError("The value of N should be an integer larger than zero. Please try again.")

    if N <= 0:
        raise ValueError("The value of N should be an integer larger than zero. Please try again.")

    return N


def validation_lag(lag):
    """ Validate lag.

    Parameters
    ----------
    lag : `int`
        time lag.

    Returns
    -------
    lag : `int`
        output right lag.

    Raises
    ------
    ValueError
        if lag is not an integer or out of range.
    """
    try:
        int(lag)
    except Exception:
        raise ValueError("The value of lag should be an integer larger than zero. Please try again.")

    if lag <= 0:
        raise ValueError("The value of lag should be an integer larger than zero. Please try again.")

    return lag


def validation_N_lag(N, lag):
    """ Validate N and lag.

    Parameters
    ----------
    N : `int`
        time step of the generated series.
    lag : `int`
        time lag.

    Raises
    ------
    ValueError
        if N is smaller or equal to lag.
    """
    if N <= lag:
        raise ValueError("The value of lag should be smaller than N. Please try again.")


def validation_num_exp(num_exp):
    """ Validate num_exp.

    Parameters
    ----------
    num_exp : `int`
        number of experiments.

    Returns
    -------
    num_exp : `int`
        output the right num_exp.

    Raises
    ------
    ValueError
        if num_exp is not an integer or out of range.
    """
    try:
        int(num_exp)
    except Exception:
        raise ValueError("The number of experiments should be an integer larger than zero. Please try again.")

    if num_exp <= 0:
        raise ValueError("The number of experiments should be an integer larger than zero. Please try again.")

    return num_exp


def validation_XY(initial_value):
    """ Validate initial value of X or Y.

    Parameters
    ----------
    initial_value : `float`
        initial value of X or Y.

    Returns
    -------
    initial_value : `float`
        output the right initial value.

    Raises
    ------
    ValueError
        if initial_value is not a number or out of range.
    """
    try:
        float(initial_value)
    except Exception:
        raise ValueError("The initial value should be a float in range of (0,1). Please try again.")

    if initial_value <= 0 or initial_value >= 1:
        raise ValueError("The initial value should be a float in range of (0,1). Please try again.")

    return initial_value





