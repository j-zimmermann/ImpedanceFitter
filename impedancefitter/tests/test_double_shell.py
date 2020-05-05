import numpy as np
from impedancefitter.double_shell import double_shell_model
omega = 2. * np.pi * np.logspace(0, 12, num=30)

# for cole_cole_model
c0 = 1.

em = 10.
km = 1e-6

kcp = 0.5
ecp = 60

kmed = .1
emed = 80

p = 0.1

dm = 5e-9
Rc = 5e-6

# for double shell
Rn = 0.8 * Rc
ene = 15.0
kne = 1e-6
knp = .5
enp = 120.0
dn = 40e-9


def test_double_shell_type():
    Z = double_shell_model(omega, km, em, kcp, ecp, ene, kne, knp, enp, kmed,
                           emed, p, c0, dm, Rc, dn, Rn)
    assert isinstance(Z, np.ndarray)


def test_double_shell_shape():
    Z = double_shell_model(omega, km, em, kcp, ecp, ene, kne, knp, enp, kmed,
                           emed, p, c0, dm, Rc, dn, Rn)
    assert Z.shape == omega.shape
