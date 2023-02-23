import numpy as np
from impedancefitter.suspensionmodels import Lk, eps_sus_ellipsoid_MW


def test_Lk():
    Rx = 0.5
    Ry = 1.0
    Rz = 2.0

    sum = 0
    for i in range(3):
        sum += Lk(Rx, Ry, Rz, i)
    assert np.isclose(sum, 1.0)


def test_MW_ellipsoid():
    Rx = 0.5
    Ry = 1.0
    Rz = 2.0
    p = 0.1
    epsi_med = np.ones(10)
    epsi_p = np.ones(10)
    eps_sus = eps_sus_ellipsoid_MW(epsi_med, epsi_p, p, Rx, Ry, Rz)
    assert isinstance(eps_sus, np.ndarray)
