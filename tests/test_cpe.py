from impedancefitter.cpe import cpe_model, cpe_ct_model, cpe_ct_w_model
import numpy as np
omega = 2. * np.pi * np.logspace(1, 8, num=20)

k = 1e-7
alpha = 0.9
Rct = 100
Aw = 10


def test_cpe_ct_w_type():
    Z = cpe_ct_w_model(omega, k, alpha, Rct, Aw)
    assert isinstance(Z, np.ndarray)


def test_cpe_ct_w_shape():
    Z = cpe_ct_w_model(omega, k, alpha, Rct, Aw)
    assert Z.shape == omega.shape


def test_cpe_ct_type():
    Z = cpe_ct_model(omega, k, alpha, Rct)
    assert isinstance(Z, np.ndarray)


def test_cpe_ct_shape():
    Z = cpe_ct_model(omega, k, alpha, Rct)
    assert Z.shape == omega.shape


def test_cpe_type():
    Z = cpe_model(omega, k, alpha)
    assert isinstance(Z, np.ndarray)


def test_cpe_shape():
    Z = cpe_model(omega, k, alpha)
    assert Z.shape == omega.shape


def test_cpe_limit_cap():
    alpha = 1.0
    Z = cpe_model(omega, k, alpha)
    assert np.all(np.isclose(np.angle(Z, deg=True), -90))


def test_cpe_limit_R():
    alpha = 0.0
    Z = cpe_model(omega, k, alpha)
    assert np.all(np.isclose(np.angle(Z, deg=True), 0.0))
