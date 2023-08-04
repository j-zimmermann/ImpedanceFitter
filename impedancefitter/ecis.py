import numpy as np
import scipy as sp


def Z_ECIS_Lo_Ferrier(omega, aecis, Lecis, Wecis, Rb, Cm, Rm, Zn=None):
    r"""ECIS impedance model as described in Lo and Ferrier, 1998

    Parameters
    ----------
    omega: :class:`numpy.ndarray`, double
        list of frequencies
    aecis: double
        :math:`alpha` parameter, Eq. 12
    Lecis: double
        :math:`L` parameter
    Wecis: double
        :math:`W` parameter
    Rb: double
        junctional resistance :math:`R_\mathrm{b}`
    Cm: double
        membrane capacitance
    Rm: double
        membrane resistance

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----

    The model is described in [Lo1998]_.

    Thanks to Paula Respondek (Uni Rostock)
    who initially implemented the model.

    References
    ----------

    .. [Lo1998] Chun-Min Lo and Jack Ferrier (1998). Impedance analysis of fibroblastic cell layers measured by electric cell-substrate impedance sensing.
                Physical Review E, 57(6), 6982-6987.
                https://dx.doi.org/10.1103/PhysRevE.57.6982

    """
    # specific membrane impedance
    Zm = 1.0 / (1.0 / Rm - 1j * omega * Cm)
    # helper
    E = Lecis * Wecis + np.pi * Wecis**2 / 4.0
    # Eq. 12
    gamma = 2.0 * aecis / Wecis * (1.0 / Zn + 1.0 / Zm)**0.5
    # Eq. 17
    Rbrec = (4.0 * Lecis / Wecis + 2 * np.pi) / (4 * Lecis / Wecis + np.pi) * Rb
    # Eq. 18
    Rbdisc = (2.0 * Lecis / Wecis + np.pi) / (4 * Lecis / Wecis + np.pi) * Rb
    # Eq. 16
    res = 1 / (Zn + Zm) * (1. + (Lecis * Wecis / E) * 2.0 * Zm / Zn / (gamma * Wecis * (1. / np.tanh(0.5 * gamma * Wecis)) + 2.0 * Rbrec * (1.0 / Zn + 1.0 / Zm))
                           + (np.pi * Wecis**2 / 4.0 / E) * 2.0 * Zm / Zn / (0.5 * gamma * Wecis * sp.special.iv(0, 0.5 * gamma * Wecis) / sp.special.iv(1, 0.5 * gamma * Wecis) + 2 * Rbdisc * (1.0 / Zn + 1.0 / Zm)))
    return 1.0 / res
