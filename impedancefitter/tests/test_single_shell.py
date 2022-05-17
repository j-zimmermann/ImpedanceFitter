import numpy as np
from impedancefitter.single_shell import single_shell_model, single_shell_bh_model, eps_cell_single_shell
from impedancefitter import get_equivalent_circuit_model
from scipy.constants import epsilon_0 as e0
from impedancefitter.suspensionmodels import eps_sus_MW, bhcubic_eps_model


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

parameters = {"c0": c0,
              "em": em,
              "km": km / 1e-6,
              "ecp": ecp,
              "kcp": kcp,
              "emed": emed,
              "kmed": kmed,
              "dm": dm,
              "Rc": Rc,
              "p": p
              }


def test_single_shell_type():
    Z = single_shell_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, dm, Rc)
    assert isinstance(Z, np.ndarray)


def test_single_shell_shape():
    Z = single_shell_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, dm, Rc)
    assert Z.shape == omega.shape


def test_single_shell_bh_type():
    Z = single_shell_bh_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, dm, Rc)
    assert isinstance(Z, np.ndarray)


def test_single_shell_bh_shape():
    Z = single_shell_bh_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, dm, Rc)
    assert Z.shape == omega.shape


def test_single_shell_impedance():
    model = "SingleShell"
    ecm = get_equivalent_circuit_model(model)
    Z = ecm.eval(omega=omega, **parameters)
    epsi_cell = eps_cell_single_shell(omega, km, em, kcp, ecp, dm, Rc)
    epsi_med = emed - 1j * kmed / (e0 * omega)
    esus = eps_sus_MW(epsi_med, epsi_cell, p)

    epsctest = 1. / (1j * omega * Z * c0 * 1e-12)

    assert np.all(np.isclose(esus, epsctest))


def test_single_shell_bh_impedance():
    model = "SingleShellBH"
    ecm = get_equivalent_circuit_model(model)
    Z = ecm.eval(omega=omega, **parameters)
    epsi_cell = eps_cell_single_shell(omega, km, em, kcp, ecp, dm, Rc)
    epsi_med = emed - 1j * kmed / (e0 * omega)
    esus = bhcubic_eps_model(epsi_med, epsi_cell, p)

    epsctest = 1. / (1j * omega * Z * c0 * 1e-12)

    assert np.all(np.isclose(esus, epsctest))
