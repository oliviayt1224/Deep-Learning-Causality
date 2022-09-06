import pytest
from DLcausality.functions.TE_class_functions import *


@pytest.mark.parametrize('T', [0, -1, -2, "one"])
def test_T(T):
    with pytest.raises(ValueError) as error:
        cwp = TE_cwp(T, N=100, alpha=0.5, lag=5, seed1=None, seed2=None)
        cwp.data_generation()
    assert error.value.args[0] == "The value of T should be a number larger than zero. Please try again."


@pytest.mark.parametrize('N', [0, -1, "hundred"])
def test_N(N):
    with pytest.raises(ValueError) as error:
        cwp = TE_cwp(T=1, N=N, alpha=0.5, lag=5, seed1=None, seed2=None)
        cwp.data_generation()
    assert error.value.args[0] == "The value of N should be an integer larger than zero. Please try again."


@pytest.mark.parametrize('lag', [0, -1, "five", "1.1"])
def test_lag(lag):
    with pytest.raises(ValueError) as error:
        cwp = TE_cwp(T=1, N=100, alpha=0.5, lag=lag, seed1=None, seed2=None)
        cwp.data_generation()
    assert error.value.args[0] == "The value of lag should be an integer larger than zero. Please try again."


@pytest.mark.parametrize('N,lag', [(4, 5), (5, 5)])
def test_N_lag(N, lag):
    with pytest.raises(ValueError) as error:
        cwp = TE_cwp(T=1, N=N, alpha=0.5, lag=lag, seed1=None, seed2=None)
        cwp.data_generation()
    assert error.value.args[0] == "The value of lag should be smaller than N. Please try again."


@pytest.mark.parametrize('alpha', [-1, 1.1, "half"])
def test_alpha(alpha):
    with pytest.raises(ValueError) as error:
        cwp = TE_cwp(T=1, N=100, alpha=alpha, lag=5, seed1=None, seed2=None)
        cwp.data_generation()
    assert error.value.args[0] == "The value of the coefficient should be a number in range of [0,1]. Please try again."


@pytest.mark.parametrize('beta', [-1, 1.1, "half"])
def test_beta(beta):
    with pytest.raises(ValueError) as error:
        twp = TE_twp(T=1, N=100, alpha=0.5, beta=beta, phi=0.5, lag=5, seed1=None, seed2=None)
        twp.data_generation()
    assert error.value.args[0] == "The value of the coefficient should be a number in range of [0,1]. Please try again."


@pytest.mark.parametrize('phi', [-1, 1.1, "half"])
def test_phi(phi):
    with pytest.raises(ValueError) as error:
        twp = TE_twp(T=1, N=100, alpha=0.5, beta=0.5, phi=phi, lag=5, seed1=None, seed2=None)
        twp.data_generation()
    assert error.value.args[0] == "The value of the coefficient should be a number in range of [0,1]. Please try again."


@pytest.mark.parametrize('epsilon', [-1, 1.1, "half"])
def test_epsilon(epsilon):
    with pytest.raises(ValueError) as error:
        clm = TE_clm(X=0.1, Y=0.1, T=1, N=100, alpha=0.5, epsilon=epsilon)
        clm.data_generation()
    assert error.value.args[0] == "The value of the coefficient should be a number in range of [0,1]. Please try again."


@pytest.mark.parametrize('X,Y', [(1, 0), (0, 0.5), ("half", 0.5)])
def test_XY(X,Y):
    with pytest.raises(ValueError) as error:
        clm = TE_clm(X=X, Y=Y, T=1, N=100, alpha=0.5, epsilon=0.5)
        clm.data_generation()
    assert error.value.args[0] == "The initial value should be a float in range of (0,1). Please try again."


@pytest.mark.parametrize('num_exp', [0, -1, "hundred", "100.1"])
def test_num_exp(num_exp):
    with pytest.raises(ValueError) as error:
        validation_num_exp(num_exp)
    assert error.value.args[0] == "The number of experiments should be an integer larger than zero. Please try again."


@pytest.mark.parametrize('T', [1, 1, 1, 1, 1])
def test_linear_TE_cwp(T):
    cwp = TE_cwp(T, N=200, alpha=0.5, lag=5, seed1=None, seed2=None)
    cwp.data_generation()
    TE = cwp.linear_TE_XY(cwp.dataset)
    assert TE > 0


@pytest.mark.parametrize('T', [1, 1, 1, 1, 1])
def test_nonlinear_TE_cwp(T):
    cwp = TE_cwp(T, N=200, alpha=0.5, lag=5, seed1=None, seed2=None)
    cwp.data_generation()
    X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(cwp.dataset)
    TE = cwp.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir)
    assert TE > 0


@pytest.mark.parametrize('X,Y', [(random.random(), random.random()), (random.random(), random.random())])
def test_nonlinear_TE_clm(X,Y):
    clm = TE_clm(X, Y, T=1, N=1000, alpha=0.4, epsilon=0.9)
    clm.data_generation()
    X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(clm.dataset)
    TE = clm.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir)
    assert TE > 0


@pytest.mark.parametrize('T', [1, 1, 1, 1, 1])
def test_linear_TE_twp(T):
    twp = TE_twp(T, N=200, alpha=0.5, phi=0.5, beta=0.5, lag=5, seed1=None, seed2=None)
    twp.data_generation()
    TE, TE_con = twp.linear_TE_XYZ(twp.dataset)
    assert TE > 0 and TE_con > 0


@pytest.mark.parametrize('T', [1, 1, 1, 1, 1])
def test_linear_TE_twp(T):
    twp = TE_twp(T, N=200, alpha=0.5, phi=0.5, beta=0.5, lag=5, seed1=None, seed2=None)
    twp.data_generation()
    X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(twp.dataset)
    TE = twp.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir)

    X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XYZ_XY(twp.dataset)
    TE_con = twp.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir)
    assert TE > 0 and TE_con > 0


@pytest.mark.parametrize('X,Y', [(0.4, 0.4)])
def test_linear_TE_tlm(X,Y):
    tlm = TE_tlm(X, Y, T=1, N=500, alpha=0.4, epsilon=0.9)
    tlm.data_generation()
    TE, TE_con = tlm.linear_TE_XYZ(tlm.dataset)
    assert TE > 0 and TE_con > 0


@pytest.mark.parametrize('X,Y', [(random.random(), random.random()), (random.random(), random.random())])
def test_nonlinear_TE_tlm(X,Y):
    tlm = TE_tlm(X, Y, T=1, N=500, alpha=0.4, epsilon=0.9)
    tlm.data_generation()
    X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(tlm.dataset)
    TE = tlm.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir)

    X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XYZ_XY(tlm.dataset)
    TE_con = tlm.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir)
    assert TE > 0 and TE_con > 0

