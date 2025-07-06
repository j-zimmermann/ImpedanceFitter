import numpy as np
from scipy.constants import epsilon_0 as e0

from impedancefitter.double_shell import eps_cell_double_shell
from impedancefitter.double_shell_wall import eps_cell_double_shell_wall

omega = 2.0 * np.pi * np.logspace(0, 12, num=30)

# for double_shell_model
c0 = 1.0

em = 10.0
km = 1e-6

kcp = 0.5
ecp = 60

kmed = 0.1
emed = 80

p = 0.1

dm = 5e-9
Rc = 5e-6

dw = 100e-9
ew = 10
kw = 0.1

# for double shell
Rn = 0.8 * Rc
ene = 15.0
kne = 1e-6
knp = 0.5
enp = 120.0
dn = 40e-9


def test_eps_cell_wall_double_shell_type():
    """Test return type."""
    eps = eps_cell_double_shell_wall(
        omega, km, em, kcp, ecp, kne, ene, knp, enp, kw, ew, dm, Rc, dn, Rn, dw
    )
    assert isinstance(eps, np.ndarray)


def test_eps_cell_wall_double_shell__shape():
    """Test return shape."""
    eps = eps_cell_double_shell_wall(
        omega, km, em, kcp, ecp, kne, ene, knp, enp, kw, ew, dm, Rc, dn, Rn, dw
    )
    assert eps.shape == omega.shape


def test_eps_cell_wall_double_shell_equivalence():
    """Test equivalence to double-shell model for
    extremely small cell wall.
    """
    # set dw to very small value
    dw = 1e-12
    eps1 = eps_cell_double_shell_wall(
        omega, km, em, kcp, ecp, kne, ene, knp, enp, kw, ew, dm, Rc, dn, Rn, dw
    )
    eps2 = eps_cell_double_shell(
        omega, km, em, kcp, ecp, kne, ene, knp, enp, dm, Rc, dn, Rn
    )
    # numerically not suitable
    # assert np.all(np.isclose(eps1, eps2, rtol=1e-4))
    assert np.all(np.isclose(eps1.real, eps2.real))
    assert np.all(
        np.isclose(-eps1.imag * e0 * omega, -eps2.imag * e0 * omega, rtol=1e-3)
    )
