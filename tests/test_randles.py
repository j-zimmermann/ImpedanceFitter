import numpy as np
from impedancefitter import get_equivalent_circuit_model
from impedancefitter.randles import Z_randles, Z_randles_CPE

omega = 2.0 * np.pi * np.logspace(1, 8, num=20)

# for randles
Rct = 100.0
Rs = 100.0
Aw = 10.0
C0 = 1e-6

# for randles_CPE
k = 1e-5
alpha = 0.6

model1 = "R_s + parallel(R_ct + W, C)"
model2 = "R_s + parallel(R_ct + W, CPE)"


def test_randles_type():
    """Test return type."""
    Z = Z_randles(omega, Rct, Rs, Aw, C0)
    assert isinstance(Z, np.ndarray)


def test_randles_shape():
    """Test return shape."""
    Z = Z_randles(omega, Rct, Rs, Aw, C0)
    assert Z.shape == omega.shape


def test_randles_CPE_type():
    """Test return type."""
    Z = Z_randles_CPE(omega, Rct, Rs, Aw, k, alpha)
    assert isinstance(Z, np.ndarray)


def test_randles_CPE_shape():
    """Test return shape."""
    Z = Z_randles_CPE(omega, Rct, Rs, Aw, k, alpha)
    assert Z.shape == omega.shape


def test_randles_model():
    """Test against manual implementation."""
    Z = Z_randles(omega, Rct, Rs, Aw, C0)
    m = get_equivalent_circuit_model(model1)
    Z_m = m.eval(omega=omega, ct_R=Rct, s_R=Rs, C=C0, Aw=Aw)
    assert np.all(np.isclose(Z, Z_m))


def test_randles_CPE_model():
    """Test against manual implementation."""
    Z = Z_randles_CPE(omega, Rct, Rs, Aw, k, alpha)
    m = get_equivalent_circuit_model(model2)
    Z_m = m.eval(omega=omega, ct_R=Rct, s_R=Rs, k=k, alpha=alpha, Aw=Aw)
    assert np.all(np.isclose(Z, Z_m))
