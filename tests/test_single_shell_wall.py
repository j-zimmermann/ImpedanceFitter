import numpy as np
from impedancefitter.single_shell import eps_cell_single_shell
from impedancefitter.single_shell_wall import eps_cell_single_shell_wall
from scipy.constants import epsilon_0 as e0

omega = 2.0 * np.pi * np.logspace(0, 12, num=30)

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
Rc = 5e-6

dw = 100e-9
ew = 10
kw = 0.1


def test_eps_cell_wall_single_shell_type():
    """Test return type."""
    eps = eps_cell_single_shell_wall(omega, km, em, kcp, ecp, kw, ew, dm, Rc, dw)
    assert isinstance(eps, np.ndarray)


def test_eps_cell_wall_single_shell__shape():
    """Test return shape."""
    eps = eps_cell_single_shell_wall(omega, km, em, kcp, ecp, kw, ew, dm, Rc, dw)
    assert eps.shape == omega.shape


def test_eps_cell_wall_single_shell_equivalence():
    """Check equivalence with single-shell model for
    infinitely thin cell wall.
    """
    # set dw to very small value
    dw = 1e-12
    eps1 = eps_cell_single_shell_wall(omega, km, em, kcp, ecp, kw, ew, dm, Rc, dw)
    eps2 = eps_cell_single_shell(omega, km, em, kcp, ecp, dm, Rc)
    # assert np.all(np.isclose(eps1, eps2, rtol=1e-4))
    assert np.all(np.isclose(eps1.real, eps2.real))
    assert np.all(
        np.isclose(-eps1.imag * e0 * omega, -eps2.imag * e0 * omega, rtol=1e-3)
    )
