from impedancefitter.RC import rc_model, RC_model, drc_model, rc_tau_model
from impedancefitter.cole_cole import cole_cole_R_model
from scipy.constants import epsilon_0 as e0
import numpy as np
omega = 2. * np.pi * np.logspace(1, 8, num=20)

# for RCfull
Rd = 100
Cd = 1e-6

# for Rctau
tauk = Rd * Cd

# for RC
c0 = 1.0
kdc = e0 / (Rd * c0 * 1e-12)
eps = Cd / (c0 * 1e-12)

# for DRC
RE = 100
tauE = 1.0
alpha = 0.9
beta = 1.0

# for cole_cole:
Rinf = 0
tau = tauE
a = 1. - alpha
R0 = RE


def test_RCfull_type():
    Z = RC_model(omega, Rd, Cd)
    assert isinstance(Z, np.ndarray)


def test_RCfull_shape():
    Z = RC_model(omega, Rd, Cd)
    assert Z.shape == omega.shape


def test_RC_type():
    Z = rc_model(omega, c0, kdc, eps)
    assert isinstance(Z, np.ndarray)


def test_RC_shape():
    Z = rc_model(omega, c0, kdc, eps)
    assert Z.shape == omega.shape


def test_DRC_type():
    Z = drc_model(omega, RE, tauE, alpha, beta)
    assert isinstance(Z, np.ndarray)


def test_DRC_shape():
    Z = drc_model(omega, RE, tauE, alpha, beta)
    assert Z.shape == omega.shape


def test_equality_rc_RC():
    Z_RC = RC_model(omega, Rd, Cd)
    Z_rc = rc_model(omega, c0, kdc, eps)
    assert np.all(np.isclose(Z_RC, Z_rc))


def test_equality_drc_cole_cole():
    Z_cole = cole_cole_R_model(omega, Rinf, R0, tau, a)
    Z_drc = drc_model(omega, RE, tauE, alpha, beta)
    assert np.all(np.isclose(Z_cole, Z_drc))


def test_RCtau_type():
    Z = rc_tau_model(omega, Rd, tauk)
    assert isinstance(Z, np.ndarray)


def test_RCtau_shape():
    Z = rc_tau_model(omega, Rd, tauk)
    assert Z.shape == omega.shape


def test_equality_rc_rc_tau():
    Z_rc = RC_model(omega, Rd, Cd)
    Z_tau = rc_tau_model(omega, Rd, tauk)
    assert np.all(np.isclose(Z_tau, Z_rc))
