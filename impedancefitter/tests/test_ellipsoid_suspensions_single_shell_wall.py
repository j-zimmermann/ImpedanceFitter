import numpy as np
from impedancefitter.single_shell_wall_ellipsoid import single_shell_wall_ellipsoid_model, eps_cell_single_shell_wall_ellipsoid
from impedancefitter.single_shell_wall import single_shell_wall_model, eps_cell_single_shell_wall


# for single_shell_wall_model
c0 = 1.

em = 10.
km = 1e-6

kcp = 0.5
ecp = 60

kmed = .1
emed = 80

kw = 0.2
ew = 40

p = 0.1

dm = 5e-9
dw = 50e-9
Rcx = 5
Rcy = 2.5
Rcz = 4
omega = 2. * np.pi * np.logspace(0, 12, num=30)


def test_single_shell_wall_ellipsoid_impedance():
    Z = single_shell_wall_ellipsoid_model(omega, km * 1e6, em, kcp, ecp, kmed, emed, kw, ew, p, c0, dm, dw, Rcx, Rcy, Rcz)
    assert isinstance(Z, np.ndarray)


def test_single_shell_wall_ellipsoid_permittivity():
    eps = eps_cell_single_shell_wall_ellipsoid(omega, km, em, kcp, ecp, kw, ew, dm, dw, Rcx, Rcy, Rcz)
    checks = []
    for i in range(3):
        checks.append(isinstance(eps[i], np.ndarray))
    assert np.all(checks)


def test_equivalence_sphere_ellipsoid_impedance():
    Rc = 5.0e-6
    Rcx = Rcy = Rcz = 5.0
    # account for unit of km
    Z_sphere = single_shell_wall_model(omega, km * 1e6, em, kcp, ecp, kw, ew, kmed, emed, p, c0, dm, Rc, dw)
    Z_ellipsoid = single_shell_wall_ellipsoid_model(omega, km * 1e6, em, kcp, ecp, kmed, emed, kw, ew, p, c0, dm * 1e6, dw * 1e6, Rcx, Rcy, Rcz)
    assert np.all(np.isclose(Z_sphere, Z_ellipsoid))


def test_equivalence_sphere_ellipsoid_permittivity():
    Rc = 5.0e-6
    Rcx = Rcy = Rcz = 5.0
    eps_sphere = eps_cell_single_shell_wall(omega, km, em, kcp, ecp, kw, ew, dm, Rc, dw)
    eps_ellipsoid = eps_cell_single_shell_wall_ellipsoid(omega, km, em, kcp, ecp, kw, ew, dm * 1e6, dw * 1e6, Rcx, Rcy, Rcz)
    checks = []
    for i in range(3):
        checks.append(np.all(np.isclose(eps_sphere, eps_ellipsoid[i])))
    assert np.all(checks)


def test_equivalence_ellipsoid_ellipsoid_impedance():
    Rcx = 5.0
    Rcy = 10.0
    Rcz = 2.0
    # account for unit of km
    Z_ellipsoid1 = single_shell_wall_ellipsoid_model(omega, km * 1e6, em, kcp, ecp, kmed, emed, kw, ew, p, c0, dm * 1e6, dw * 1e6, Rcx, Rcy, Rcz)
    Rcx = 10.0
    Rcy = 5.0
    Rcz = 2.0
    Z_ellipsoid2 = single_shell_wall_ellipsoid_model(omega, km * 1e6, em, kcp, ecp, kmed, emed, kw, ew, p, c0, dm * 1e6, dw * 1e6, Rcx, Rcy, Rcz)
    assert np.all(np.isclose(Z_ellipsoid1, Z_ellipsoid2))
