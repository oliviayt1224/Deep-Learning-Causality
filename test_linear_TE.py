import pytest
from function import *

T_parametrize = [i for i in range ( 1 , 11 , 1 )]
@pytest.mark.parametrize('T', T_parametrize)
def test_linear_TE_oupled_wiener_process(T):
    data = coupled_wiener_process(T, N = 100, alpha = 0.5, lag = 5, seed1=None, seed2=None)
    col_names = data.columns.values.tolist()
    TE = linear_TE(col_names[1], col_names[0], 5, data)
    assert TE > 0



