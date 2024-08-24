import numpy as np
from impedancefitter.cole_cole import cole_cole_model, cole_cole_R_model, raicu
from impedancefitter.utils import return_diel_properties

omega = 2.0 * np.pi * np.logspace(1, 8, num=20)

# for cole_cole_model
c0 = 1.0
el = 81.0
eh = 1.8
tau = 9.4e-3
a = 1.0 - 0.009
kdc = 1e-5

# for cole_cole_R_model
Rinf = 100
R0 = 10e3


def test_cole_cole_type():
    """Test Cole-Cole model return type."""
    Z = cole_cole_model(omega, c0, el, tau, a, kdc, eh)
    assert isinstance(Z, np.ndarray)


def test_cole_cole_shape():
    """Test Cole-Cole model return shape."""
    Z = cole_cole_model(omega, c0, el, tau, a, kdc, eh)
    assert Z.shape == omega.shape


def test_cole_cole_limits_low():
    """Test Cole-Cole model low-frequency limit."""
    Z = cole_cole_model(omega, c0, el, tau, a, kdc, eh)
    eps, _ = return_diel_properties(omega, Z, c0 * 1e-12)
    assert np.isclose(eps[0], el), str(eps[0])


def test_cole_cole_R_type():
    """Test Cole-Cole model return type."""
    Z = cole_cole_R_model(omega, Rinf, R0, tau, a)
    assert isinstance(Z, np.ndarray)


def test_cole_cole_R_shape():
    """Test Cole-Cole model return shape."""
    Z = cole_cole_R_model(omega, Rinf, R0, tau, a)
    assert Z.shape == omega.shape


def test_cole_cole_R_limits_low():
    """Test Cole-Cole model low-frequency limit."""
    Z = cole_cole_R_model(omega, Rinf, R0, tau, a)
    assert np.isclose(Z[0], R0), np.isclose(Z[-1], Rinf)


def test_raicu():
    """Test Raicu model equivalence to Cole-Cole model."""
    Zcole = cole_cole_model(omega, c0, el, tau, a, kdc, eh)
    Zraicu = raicu(omega, c0, eh, (el - eh) / 1e3, tau * 1e-3, a, 1.0, 1.0, kdc)
    assert np.all(np.isclose(Zcole, Zraicu))
