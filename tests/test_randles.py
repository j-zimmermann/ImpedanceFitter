from impedancefitter.randles import Z_randles_CPE, Z_randles
from impedancefitter import get_equivalent_circuit_model

import numpy as np
omega = 2. * np.pi * np.logspace(1, 8, num=20)

# for randles
Rct = 100.
Rs = 100.
Aw = 10.
C0 = 1e-6

# for randles_CPE
k = 1e-5
alpha = 0.6

model1 = 'R_s + parallel(R_ct + W, C)'
model2 = 'R_s + parallel(R_ct + W, CPE)'


def test_randles_type():
    Z = Z_randles(omega, Rct, Rs, Aw, C0)
    assert isinstance(Z, np.ndarray)


def test_randles_shape():
    Z = Z_randles(omega, Rct, Rs, Aw, C0)
    assert Z.shape == omega.shape


def test_randles_CPE_type():
    Z = Z_randles_CPE(omega, Rct, Rs, Aw, k, alpha)
    assert isinstance(Z, np.ndarray)


def test_randles_CPE_shape():
    Z = Z_randles_CPE(omega, Rct, Rs, Aw, k, alpha)
    assert Z.shape == omega.shape


def test_randles_model():
    Z = Z_randles(omega, Rct, Rs, Aw, C0)
    m = get_equivalent_circuit_model(model1)
    Z_m = m.eval(omega=omega, ct_R=Rct, s_R=Rs,
                 C=C0, Aw=Aw)
    assert np.all(np.isclose(Z, Z_m))


def test_randles_CPE_model():
    Z = Z_randles_CPE(omega, Rct, Rs, Aw, k, alpha)
    m = get_equivalent_circuit_model(model2)
    Z_m = m.eval(omega=omega, ct_R=Rct, s_R=Rs,
                 k=k, alpha=alpha, Aw=Aw)
    assert np.all(np.isclose(Z, Z_m))
