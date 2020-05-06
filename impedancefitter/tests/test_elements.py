import numpy as np
from impedancefitter.elements import (Z_R, Z_C, Z_L, Z_w,
                                      Z_wo, Z_ws, Z_stray, Z_CPE,
                                      parallel)

omega = 2. * np.pi * np.logspace(1, 8, num=20)

R = 100
alpha = .5


def test_Z_R_type():
    Z = Z_R(omega, R)
    assert isinstance(Z, np.ndarray)


def test_Z_R_shape():
    Z = Z_R(omega, R)
    assert Z.shape == omega.shape


def test_Z_L_type():
    Z = Z_L(omega, R)
    assert isinstance(Z, np.ndarray)


def test_Z_L_shape():
    Z = Z_L(omega, R)
    assert Z.shape == omega.shape


def test_Z_CPE_type():
    Z = Z_CPE(omega, R, alpha)
    assert isinstance(Z, np.ndarray)


def test_Z_CPE_shape():
    Z = Z_CPE(omega, R, alpha)
    assert Z.shape == omega.shape


def test_Z_C_type():
    Z = Z_C(omega, R)
    assert isinstance(Z, np.ndarray)


def test_Z_C_shape():
    Z = Z_C(omega, R)
    assert Z.shape == omega.shape


def test_Z_w_type():
    Z = Z_w(omega, R)
    assert isinstance(Z, np.ndarray)


def test_Z_w_shape():
    Z = Z_w(omega, R)
    assert Z.shape == omega.shape


def test_Z_wo_type():
    Z = Z_wo(omega, R, alpha)
    assert isinstance(Z, np.ndarray)


def test_Z_wo_shape():
    Z = Z_wo(omega, R, alpha)
    assert Z.shape == omega.shape


def test_Z_ws_type():
    Z = Z_ws(omega, R, alpha)
    assert isinstance(Z, np.ndarray)


def test_Z_ws_shape():
    Z = Z_ws(omega, R, alpha)
    assert Z.shape == omega.shape


def test_Z_stray_type():
    Z = Z_stray(omega, R)
    assert isinstance(Z, np.ndarray)


def test_Z_stray_shape():
    Z = Z_stray(omega, R)
    assert Z.shape == omega.shape


def test_parallel_shape():
    Z1 = Z_R(omega, R)
    Z2 = Z_R(omega, R)
    Z = parallel(Z1, Z2)
    assert Z.shape == omega.shape


def test_parallel_type():
    Z1 = Z_R(omega, R)
    Z2 = Z_R(omega, R)
    Z = parallel(Z1, Z2)

    assert isinstance(Z, np.ndarray)


def test_parallel():
    Z1 = Z_R(omega, R)
    Z = parallel(Z1, Z1)
    assert np.all(np.isclose(Z, 0.5 * Z1))
