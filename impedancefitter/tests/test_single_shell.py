import numpy as np
from impedancefitter.single_shell import single_shell_model


omega = 2. * np.pi * np.logspace(0, 12, num=30)

# for single_shell_model
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
Rn = 1.0
ene = 1.0
kne = 0.0
knp = 0.0
enp = 0.0
dn = 0.0


def test_single_shell_type():
    Z = single_shell_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, dm, Rc)
    assert isinstance(Z, np.ndarray)


def test_single_shell_shape():
    Z = single_shell_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, dm, Rc)
    assert Z.shape == omega.shape
