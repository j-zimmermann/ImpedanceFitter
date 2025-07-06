import numpy as np

from impedancefitter.cpe import cpe_ct_model, cpe_ct_w_model, cpe_model

omega = 2.0 * np.pi * np.logspace(1, 8, num=20)

k = 1e-7
alpha = 0.9
Rct = 100
Aw = 10


def test_cpe_ct_w_type():
    """Test CPE return type."""
    Z = cpe_ct_w_model(omega, k, alpha, Rct, Aw)
    assert isinstance(Z, np.ndarray)


def test_cpe_ct_w_shape():
    """Test CPE return shape."""
    Z = cpe_ct_w_model(omega, k, alpha, Rct, Aw)
    assert Z.shape == omega.shape


def test_cpe_ct_type():
    """Test CPE return type."""
    Z = cpe_ct_model(omega, k, alpha, Rct)
    assert isinstance(Z, np.ndarray)


def test_cpe_ct_shape():
    """Test CPE return shape."""
    Z = cpe_ct_model(omega, k, alpha, Rct)
    assert Z.shape == omega.shape


def test_cpe_type():
    """Test CPE return type."""
    Z = cpe_model(omega, k, alpha)
    assert isinstance(Z, np.ndarray)


def test_cpe_shape():
    """Test CPE return shape."""
    Z = cpe_model(omega, k, alpha)
    assert Z.shape == omega.shape


def test_cpe_limit_cap():
    """Test CPE limit: equal to capacitor for a=1."""
    alpha = 1.0
    Z = cpe_model(omega, k, alpha)
    assert np.all(np.isclose(np.angle(Z, deg=True), -90))


def test_cpe_limit_R():
    """Test CPE limit: equal to resistor for a=0."""
    alpha = 0.0
    Z = cpe_model(omega, k, alpha)
    assert np.all(np.isclose(np.angle(Z, deg=True), 0.0))
