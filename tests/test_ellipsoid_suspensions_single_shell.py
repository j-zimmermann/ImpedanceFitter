import numpy as np

from impedancefitter.single_shell import eps_cell_single_shell, single_shell_model
from impedancefitter.single_shell_ellipsoid import (
    eps_cell_single_shell_ellipsoid,
    single_shell_ellipsoid_model,
)
from impedancefitter.suspensionmodels import Lk, eps_sus_ellipsoid_MW

# for single_shell_model
c0 = 1.0

em = 10.0
km = 1e-6

kcp = 0.5
ecp = 60

kmed = 0.1
emed = 80

p = 0.1

dm = 5e-9
Rcx = 5
Rcy = 2.5
Rcz = 4
omega = 2.0 * np.pi * np.logspace(0, 12, num=30)


def test_Lk():
    """Test against analytical expression."""
    Rx = 0.5
    Ry = 1.0
    Rz = 2.0

    sum = 0
    for i in range(3):
        sum += Lk(Rx, Ry, Rz, i)
    assert np.isclose(sum, 1.0)


def test_MW_ellipsoid():
    """Test return type."""
    Rx = 0.5
    Ry = 1.0
    Rz = 2.0
    p = 0.1
    epsi_med = np.ones(10)
    epsi_p = np.ones(10)
    eps_sus = eps_sus_ellipsoid_MW(epsi_med, epsi_p, epsi_p, epsi_p, p, Rx, Ry, Rz)
    assert isinstance(eps_sus, np.ndarray)


def test_single_shell_ellipsoid_impedance():
    """Test return type."""
    Z = single_shell_ellipsoid_model(
        omega, km * 1e6, em, kcp, ecp, kmed, emed, p, c0, dm * 1e6, Rcx, Rcy, Rcz
    )
    assert isinstance(Z, np.ndarray)


def test_single_shell_ellipsoid_permittivity():
    """Test return type."""
    eps = eps_cell_single_shell_ellipsoid(
        omega, km, em, kcp, ecp, dm * 1e6, Rcx, Rcy, Rcz
    )
    checks = []
    for i in range(3):
        checks.append(isinstance(eps[i], np.ndarray))
    assert np.all(checks)


def test_equivalence_sphere_ellipsoid_impedance():
    """Check that sphere and ellipsoid code yield
    the same result.
    """
    Rc = 5.0e-6
    Rcx = Rcy = Rcz = 5.0
    # account for unit of km
    Z_sphere = single_shell_model(
        omega, km * 1e6, em, kcp, ecp, kmed, emed, p, c0, dm, Rc
    )
    Z_ellipsoid = single_shell_ellipsoid_model(
        omega, km * 1e6, em, kcp, ecp, kmed, emed, p, c0, dm * 1e6, Rcx, Rcy, Rcz
    )
    assert np.all(np.isclose(Z_sphere, Z_ellipsoid))


def test_equivalence_sphere_ellipsoid_permittivity():
    """Check that sphere and ellipsoid code yield
    the same result.
    """
    Rc = 5.0e-6
    Rcx = Rcy = Rcz = 5.0
    eps_sphere = eps_cell_single_shell(omega, km, em, kcp, ecp, dm, Rc)
    eps_ellipsoid = eps_cell_single_shell_ellipsoid(
        omega, km, em, kcp, ecp, dm * 1e6, Rcx, Rcy, Rcz
    )
    checks = []
    for i in range(3):
        checks.append(np.all(np.isclose(eps_sphere, eps_ellipsoid[i])))
    assert np.all(checks)


def test_equivalence_ellipsoid_ellipsoid_impedance():
    """Check equivalence for flipped axis."""
    Rcx = 5.0
    Rcy = 10.0
    Rcz = 2.0
    # account for unit of km
    Z_ellipsoid1 = single_shell_ellipsoid_model(
        omega, km * 1e6, em, kcp, ecp, kmed, emed, p, c0, dm * 1e6, Rcx, Rcy, Rcz
    )
    Rcx = 10.0
    Rcy = 5.0
    Rcz = 2.0
    Z_ellipsoid2 = single_shell_ellipsoid_model(
        omega, km * 1e6, em, kcp, ecp, kmed, emed, p, c0, dm * 1e6, Rcx, Rcy, Rcz
    )
    assert np.all(np.isclose(Z_ellipsoid1, Z_ellipsoid2))
