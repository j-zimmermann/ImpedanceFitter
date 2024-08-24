import numpy as np
from impedancefitter.cpe import cpe_ct_model, cpe_ct_w_model
from impedancefitter.utils import get_equivalent_circuit_model

omega = 2.0 * np.pi * np.logspace(1, 8, num=20)


def test_cpe_RCT():
    """Test CPE in parallel with charge-transfer resistance."""
    k = 1e-7
    alpha = 0.9
    Rct = 100

    model = "parallel(CPE, R)"
    m = get_equivalent_circuit_model(model)
    Z_lmfit = m.eval(omega=omega, k=k, alpha=alpha, R=Rct)
    Z = cpe_ct_model(omega, k, alpha, Rct)
    assert np.all(np.isclose(Z, Z_lmfit))


def test_cpe_RCT_W():
    """Test CPE in parallel with charge-transfer resistance
    and Warburg element.
    """
    k = 1e-7
    alpha = 0.9
    Rct = 100
    Aw = 10

    model = "parallel(CPE, R + W)"
    m = get_equivalent_circuit_model(model)
    Z_lmfit = m.eval(omega=omega, k=k, alpha=alpha, R=Rct, Aw=Aw)
    Z = cpe_ct_w_model(omega, k, alpha, Rct, Aw)
    assert np.all(np.isclose(Z, Z_lmfit))
