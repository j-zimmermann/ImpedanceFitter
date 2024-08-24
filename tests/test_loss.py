from impedancefitter.loss import Z_loss, Z_in
from impedancefitter import get_equivalent_circuit_model

import numpy as np
omega = 2. * np.pi * np.logspace(1, 8, num=20)

L = 1
C = 1
R = 1


def test_loss_type():
    Z = Z_loss(omega, L, C, R)
    assert isinstance(Z, np.ndarray)


def test_loss_shape():
    Z = Z_loss(omega, L, C, R)
    assert Z.shape == omega.shape


def test_in_type():
    Z = Z_in(omega, L, R)
    assert isinstance(Z, np.ndarray)


def test_in_shape():
    Z = Z_in(omega, L, R)
    assert Z.shape == omega.shape


def test_loss_model():
    model = "parallel(R, parallel(L, C))"
    m = get_equivalent_circuit_model(model)
    Z = Z_loss(omega, L, C, R)
    Z_m = m.eval(omega=omega, L=L * 1e-9, R=R,
                 C=C * 1e-12)
    assert np.all(np.isclose(Z, Z_m))


def test_in_model():
    model = "R + L"
    m = get_equivalent_circuit_model(model)
    Z = Z_in(omega, L, R)
    Z_m = m.eval(omega=omega, L=L * 1e-9, R=R)
    assert np.all(np.isclose(Z, Z_m))
