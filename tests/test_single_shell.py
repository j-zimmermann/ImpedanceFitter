import numpy as np
from scipy.constants import epsilon_0 as e0

from impedancefitter import get_equivalent_circuit_model
from impedancefitter.single_shell import (
    eps_cell_single_shell,
    single_shell_bh_model,
    single_shell_model,
)
from impedancefitter.suspensionmodels import bhcubic_eps_model, eps_sus_MW

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

parameters = {
    "c0": c0,
    "em": em,
    "km": km / 1e-6,
    "ecp": ecp,
    "kcp": kcp,
    "emed": emed,
    "kmed": kmed,
    "dm": dm,
    "Rc": Rc,
    "p": p,
}


def test_single_shell_type():
    """Test return type."""
    Z = single_shell_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, dm, Rc)
    assert isinstance(Z, np.ndarray)


def test_single_shell_shape():
    """Test return shape."""
    Z = single_shell_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, dm, Rc)
    assert Z.shape == omega.shape


def test_single_shell_bh_type():
    """Test return type."""
    Z = single_shell_bh_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, dm, Rc)
    assert isinstance(Z, np.ndarray)


def test_single_shell_bh_shape():
    """Test return shape."""
    Z = single_shell_bh_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, dm, Rc)
    assert Z.shape == omega.shape


def test_single_shell_impedance():
    """Test correct impedance calculation from
    complex permittivity.
    """
    model = "SingleShell"
    ecm = get_equivalent_circuit_model(model)
    Z = ecm.eval(omega=omega, **parameters)
    epsi_cell = eps_cell_single_shell(omega, km, em, kcp, ecp, dm, Rc)
    epsi_med = emed - 1j * kmed / (e0 * omega)
    esus = eps_sus_MW(epsi_med, epsi_cell, p)

    epsctest = 1.0 / (1j * omega * Z * c0 * 1e-12)

    assert np.all(np.isclose(esus, epsctest))


def test_single_shell_bh_impedance():
    """Test correct impedance calculation from
    complex permittivity.
    """
    model = "SingleShellBH"
    ecm = get_equivalent_circuit_model(model)
    Z = ecm.eval(omega=omega, **parameters)
    epsi_cell = eps_cell_single_shell(omega, km, em, kcp, ecp, dm, Rc)
    epsi_med = emed - 1j * kmed / (e0 * omega)
    esus = bhcubic_eps_model(epsi_med, epsi_cell, p)

    epsctest = 1.0 / (1j * omega * Z * c0 * 1e-12)

    assert np.all(np.isclose(esus, epsctest))
