import numpy as np
from impedancefitter.particle_suspension import particle_model, particle_bh_model
from impedancefitter.single_shell import single_shell_model, single_shell_bh_model

omega = 2. * np.pi * np.logspace(0, 12, num=30)

c0 = 1.


kmed = .1
emed = 80

p = 0.1

Rc = 5e-6
# for single_shell_model
dm = 0.
em = 1.
km = 1.

# note difference in units between both models
kp = 0.5
kcp = 0.5e-6
ecp = 60


def test_particle_type():
    Z = particle_model(omega, ecp, kp, kmed, emed, p, c0)
    assert isinstance(Z, np.ndarray)


def test_particle_shape():
    Z = particle_model(omega, ecp, kp, kmed, emed, p, c0)
    assert Z.shape == omega.shape


def test_particle_bh_type():
    Z = particle_bh_model(omega, ecp, kp, kmed, emed, p, c0)
    assert isinstance(Z, np.ndarray)


def test_particle_bh_shape():
    Z = particle_bh_model(omega, ecp, kp, kmed, emed, p, c0)
    assert Z.shape == omega.shape


def test_equivalence_particle_single_shell():
    Zp = particle_model(omega, ecp, kp, kmed, emed, p, c0)
    Zs = single_shell_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, dm, Rc)
    assert np.all(np.isclose(Zp, Zs))


def test_equivalence_particle_single_shell_bh():
    Zp = particle_bh_model(omega, ecp, kp, kmed, emed, p, c0)
    Zs = single_shell_bh_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, dm, Rc)
    assert np.all(np.isclose(Zp, Zs))
