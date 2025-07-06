import numpy as np

from impedancefitter.particle_suspension import particle_bh_model, particle_model
from impedancefitter.single_shell import single_shell_bh_model, single_shell_model

omega = 2.0 * np.pi * np.logspace(0, 12, num=30)

c0 = 1.0


kmed = 0.1
emed = 80

p = 0.1

Rc = 5e-6
# for single_shell_model
dm = 0.0
em = 1.0
km = 1.0

# note difference in units between both models
kp = 0.5
kcp = 0.5e-6
ecp = 60


def test_particle_type():
    """Test return type."""
    Z = particle_model(omega, ecp, kp, kmed, emed, p, c0)
    assert isinstance(Z, np.ndarray)


def test_particle_shape():
    """Test return shape."""
    Z = particle_model(omega, ecp, kp, kmed, emed, p, c0)
    assert Z.shape == omega.shape


def test_particle_bh_type():
    """Test return type."""
    Z = particle_bh_model(omega, ecp, kp, kmed, emed, p, c0)
    assert isinstance(Z, np.ndarray)


def test_particle_bh_shape():
    """Test return shape."""
    Z = particle_bh_model(omega, ecp, kp, kmed, emed, p, c0)
    assert Z.shape == omega.shape


def test_equivalence_particle_single_shell():
    """Check equivalence with single shell model
    for infinitely thin membrane.
    """
    Zp = particle_model(omega, ecp, kp, kmed, emed, p, c0)
    Zs = single_shell_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, dm, Rc)
    assert np.all(np.isclose(Zp, Zs))


def test_equivalence_particle_single_shell_bh():
    """Check equivalence with single shell model
    for infinitely thin membrane.
    """
    Zp = particle_bh_model(omega, ecp, kp, kmed, emed, p, c0)
    Zs = single_shell_bh_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, dm, Rc)
    assert np.all(np.isclose(Zp, Zs))
