import numpy as np
from impedancefitter.elements import (
    Z_C,
    Z_CPE,
    Z_L,
    Z_R,
    Z_stray,
    Z_w,
    Z_wo,
    Z_ws,
    parallel,
)

omega = 2.0 * np.pi * np.logspace(1, 8, num=20)

R = 100
alpha = 0.5


def test_Z_R_type():
    """Test return type."""
    Z = Z_R(omega, R)
    assert isinstance(Z, np.ndarray)


def test_Z_R_shape():
    """Test return shape."""
    Z = Z_R(omega, R)
    assert Z.shape == omega.shape


def test_Z_L_type():
    """Test return type."""
    Z = Z_L(omega, R)
    assert isinstance(Z, np.ndarray)


def test_Z_L_shape():
    """Test return shape."""
    Z = Z_L(omega, R)
    assert Z.shape == omega.shape


def test_Z_CPE_type():
    """Test return type."""
    Z = Z_CPE(omega, R, alpha)
    assert isinstance(Z, np.ndarray)


def test_Z_CPE_shape():
    """Test return shape."""
    Z = Z_CPE(omega, R, alpha)
    assert Z.shape == omega.shape


def test_Z_C_type():
    """Test return type."""
    Z = Z_C(omega, R)
    assert isinstance(Z, np.ndarray)


def test_Z_C_shape():
    """Test return shape."""
    Z = Z_C(omega, R)
    assert Z.shape == omega.shape


def test_Z_w_type():
    """Test return type."""
    Z = Z_w(omega, R)
    assert isinstance(Z, np.ndarray)


def test_Z_w_shape():
    """Test return shape."""
    Z = Z_w(omega, R)
    assert Z.shape == omega.shape


def test_Z_wo_type():
    """Test return type."""
    Z = Z_wo(omega, R, alpha)
    assert isinstance(Z, np.ndarray)


def test_Z_wo_shape():
    """Test return shape."""
    Z = Z_wo(omega, R, alpha)
    assert Z.shape == omega.shape


def test_Z_ws_type():
    """Test return type."""
    Z = Z_ws(omega, R, alpha)
    assert isinstance(Z, np.ndarray)


def test_Z_ws_shape():
    """Test return shape."""
    Z = Z_ws(omega, R, alpha)
    assert Z.shape == omega.shape


def test_Z_stray_type():
    """Test return type."""
    Z = Z_stray(omega, R)
    assert isinstance(Z, np.ndarray)


def test_Z_stray_shape():
    """Test return shape."""
    Z = Z_stray(omega, R)
    assert Z.shape == omega.shape


def test_parallel_shape():
    """Test return shape."""
    Z1 = Z_R(omega, R)
    Z2 = Z_R(omega, R)
    Z = parallel(Z1, Z2)
    assert Z.shape == omega.shape


def test_parallel_type():
    """Test return type."""
    Z1 = Z_R(omega, R)
    Z2 = Z_R(omega, R)
    Z = parallel(Z1, Z2)

    assert isinstance(Z, np.ndarray)


def test_parallel():
    """Test correctness of parallel circuit implementation."""
    Z1 = Z_R(omega, R)
    Z = parallel(Z1, Z1)
    assert np.all(np.isclose(Z, 0.5 * Z1))
