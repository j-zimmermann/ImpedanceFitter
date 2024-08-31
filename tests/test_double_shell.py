import numpy as np
from impedancefitter.double_shell import double_shell_model
from impedancefitter.single_shell import single_shell_model

omega = 2.0 * np.pi * np.logspace(0, 12, num=30)

# for cole_cole_model
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

# for double shell
Rn = 0.8 * Rc
ene = 15.0
kne = 1e-6
knp = 0.5
enp = 120.0
dn = 40e-9


def test_double_shell_type():
    """Test return type."""
    Z = double_shell_model(
        omega, km, em, kcp, ecp, ene, kne, knp, enp, kmed, emed, p, c0, dm, Rc, dn, Rn
    )
    assert isinstance(Z, np.ndarray)


def test_double_shell_shape():
    """Test return shape."""
    Z = double_shell_model(
        omega, km, em, kcp, ecp, ene, kne, knp, enp, kmed, emed, p, c0, dm, Rc, dn, Rn
    )
    assert Z.shape == omega.shape


def test_equivalence_single_double():
    """Test equivalence to single-shell model for
    extremely small nucleus.
    """
    # set radius of nucleus to zero: models should be equivalent
    Rn = 1e-12
    Z1 = double_shell_model(
        omega, km, em, kcp, ecp, ene, kne, knp, enp, kmed, emed, p, c0, dm, Rc, dn, Rn
    )
    Z2 = single_shell_model(omega, km, em, kcp, ecp, kmed, emed, p, c0, dm, Rc)
    assert np.all(np.isclose(Z1, Z2))
