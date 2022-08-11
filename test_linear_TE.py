import pytest
from TE_class_functions import *

T_parametrize = [i for i in range ( 1 , 11 , 1 )]
@pytest.mark.parametrize('T', T_parametrize)
def test_linear_TE_coupled_wiener_process(T):
    data = coupled_wiener_process(T, N = 100, alpha = 0.5, lag = 5, seed1=None, seed2=None)
    col_names = data.columns.values.tolist()
    TE = linear_TE(col_names[1], col_names[0], data, 5)
    assert TE > 0


T_parametrize = [i for i in range ( 1 , 11 , 1 )]
@pytest.mark.parametrize('T', T_parametrize)
def test_linear_TE_tenary_wiener_process(T):
    data = ternary_wiener_process(T, N=100, alpha=0.2, phi=0.5, beta=0.5, lag=5)
    col_names = data.columns.values.tolist()
    TE = linear_TE(col_names[1], col_names[0], data, 5)
    conditional_TE = conditional_linear_TE(col_names[1], col_names[0], col_names[2], data, 5)
    assert TE > 0 and conditional_TE >0

