import numpy as np
from impedancefitter.double_shell_wall_ellipsoid import double_shell_wall_ellipsoid_model, eps_cell_double_shell_wall_ellipsoid
from impedancefitter.double_shell_wall import double_shell_wall_model, eps_cell_double_shell_wall


# for double_shell_wall_model
c0 = 1.

em = 10.
km = 1e-6

kcp = 0.5
ecp = 60

kmed = .1
emed = 80

kw = 0.2
ew = 40

dw = 50e-9

p = 0.1

dm = 5e-9
Rcx = 5
Rcy = 2.5
Rcz = 4

# for double shell
Rnx = 0.8 * Rcx
Rny = 0.8 * Rcy
Rnz = 0.8 * Rcz
ene = 15.0
kne = 1e-6
knp = .5
enp = 120.0
dn = 40e-9


omega = 2. * np.pi * np.logspace(0, 12, num=30)


def test_double_shell_wall_ellipsoid_impedance():
    Z = double_shell_wall_ellipsoid_model(omega, km * 1e6, em, kcp, ecp, kmed, emed, kne * 1e3, ene, knp, enp, kw, ew, p, c0, dm * 1e6, dn * 1e6, dw * 1e6, Rcx, Rcy, Rcz, Rnx, Rny, Rnz)
    assert isinstance(Z, np.ndarray)


def test_double_shell_wall_ellipsoid_permittivity():
    eps = eps_cell_double_shell_wall_ellipsoid(omega, km, em, kcp, ecp, kne * 1e3, ene, knp, enp, kw, ew, dm, dn, dw, Rcx, Rcy, Rcz, Rnx, Rny, Rnz)
    checks = []
    for i in range(3):
        checks.append(isinstance(eps[i], np.ndarray))
    assert np.all(checks)


def test_equivalence_sphere_ellipsoid_impedance():
    Rc = 5.0e-6
    Rn = 0.8 * Rc
    Rcx = Rcy = Rcz = 5.0
    Rnx = Rny = Rnz = 0.8 * Rcx
    # account for unit of km
    Z_sphere = double_shell_wall_model(omega, km * 1e6, em, kcp, ecp, kne * 1e3, ene, knp, enp, kw, ew, kmed, emed, p, c0, dm, Rc, dn, Rn, dw)
    Z_ellipsoid = double_shell_wall_ellipsoid_model(omega, km * 1e6, em, kcp, ecp, kmed, emed, kne * 1e3, ene, knp, enp, kw, ew, p, c0, dm * 1e6, dn * 1e6, dw * 1e6, Rcx, Rcy, Rcz, Rnx, Rny, Rnz)
    assert np.all(np.isclose(Z_sphere, Z_ellipsoid))


def test_equivalence_sphere_ellipsoid_permittivity():
    Rc = 5.0e-6
    Rn = 0.8 * Rc
    Rcx = Rcy = Rcz = 5.0
    Rnx = Rny = Rnz = 0.8 * Rcx
    eps_sphere = eps_cell_double_shell_wall(omega, km, em, kcp, ecp, kne, ene, knp, enp, kw, ew, dm, Rc, dn, Rn, dw)
    eps_ellipsoid = eps_cell_double_shell_wall_ellipsoid(omega, km, em, kcp, ecp, kne, ene, knp, enp, kw, ew, dm * 1e6, dn * 1e6, dw * 1e6, Rcx, Rcy, Rcz, Rnx, Rny, Rnz)
    checks = []
    for i in range(3):
        checks.append(np.all(np.isclose(eps_sphere, eps_ellipsoid[i])))
    assert np.all(checks)


def test_equivalence_ellipsoid_ellipsoid_impedance():
    Rcx = 5.0
    Rcy = 10.0
    Rcz = 2.0
    Rnx = 0.8 * Rcx
    Rny = 0.8 * Rcy
    Rny = 0.8 * Rcy
    # account for unit of km
    Z_ellipsoid1 = double_shell_wall_ellipsoid_model(omega, km * 1e6, em, kcp, ecp, kmed, emed, kne * 1e3, ene, knp, enp, kw, ew, p, c0, dm * 1e6, dn * 1e6, dw * 1e6, Rcx, Rcy, Rcz, Rnx, Rny, Rnz)
    Rcx = 10.0
    Rcy = 5.0
    Rcz = 2.0
    Rnx = 0.8 * Rcx
    Rny = 0.8 * Rcy
    Rny = 0.8 * Rcy
    Z_ellipsoid2 = double_shell_wall_ellipsoid_model(omega, km * 1e6, em, kcp, ecp, kmed, emed, kne * 1e3, ene, knp, enp, kw, ew, p, c0, dm * 1e6, dn * 1e6, dw * 1e6, Rcx, Rcy, Rcz, Rnx, Rny, Rnz)
    assert np.all(np.isclose(Z_ellipsoid1, Z_ellipsoid2))
