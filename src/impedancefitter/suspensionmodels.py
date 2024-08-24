#    The ImpedanceFitter is a package to fit impedance spectra to
#    equivalent-circuit models using open-source software.
#
#    Copyright (C) 2021 Julius Zimmermann,
#                                   julius.zimmermann[AT]uni-rostock.de
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
from scipy.integrate import quad

from .cubic_roots import get_cubic_roots


def Lk_body(s, Rk, Rx, Ry, Rz):
    """Formula for Lk."""
    Rs = np.sqrt((Rx**2 + s) * (Ry**2 + s) * (Rz**2 + s))
    return 1.0 / ((Rk**2 + s) * Rs)


def Lk(Rx, Ry, Rz, k):
    r"""Depolarization factor for ellipsoids.

    Notes
    -----
    The depolarization factor is given as [Asami2002]_

    .. math::

        L_k = \frac{R_x R_y R_z}{2} \int_0^\infty \frac{\mathrm{d} s}{(R_k^2 + s)R_s}

    with

    .. math::

        R_s = \sqrt{(R_x^2 + s)(R_y^2 + s)(R_z^2 + s)}

    TODO: watch units!

    """
    if np.any(np.less_equal(np.array([Rx, Ry, Rz]), 1e-3)):
        raise RuntimeError(
            "Attention! The radius is small so the integral will not evaluate properly!"
        )
    Rk = None
    if k == 0:
        Rk = Rx
    elif k == 1:
        Rk = Ry
    elif k == 2:
        Rk = Rz
    else:
        raise RuntimeError("The parameter k in L_k must be in [0, 2]")
    integral = quad(Lk_body, 0, np.inf, args=(Rk, Rx, Ry, Rz))
    return 0.5 * Rx * Ry * Rz * integral[0]


def eps_sus_MW(epsi_med, epsi_p, p):
    r"""Maxwell-Wagner mixture model for dilute suspensions of spherical particles:w.


    Parameters
    ----------
    epsi_med: :class:`numpy.ndarray`, complex
        complex permittivities of medium
    epsi_p: :class:`numpy.ndarray`, complex
        complex permittivities of suspension phase (cells, particles...)
    p: double
        volume fraction

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Complex permittivity array

    Notes
    -----
    The complex permittivity of the suspension
    :math:`\varepsilon_\mathrm{sus}^\ast` is given by

    .. math::
        \varepsilon_\mathrm{sus}^\ast = \varepsilon_\mathrm{med}^\ast
        \frac{(2\varepsilon_\mathrm{med}^\ast+\varepsilon_\mathrm{p}^\ast)
        -2p(\varepsilon_\mathrm{med}^\ast-\varepsilon_\mathrm{p}^\ast)}
        {(2\varepsilon_\mathrm{med}^\ast+\varepsilon_\mathrm{p}^\ast)+
        p(\varepsilon_\mathrm{med}^\ast-\varepsilon_\mathrm{p}^\ast)}
        \enspace ,

    with :math:`\varepsilon_\mathrm{med}^\ast` being the permittivity
    of the liquid medium and :math:`\varepsilon_\mathrm{p}^\ast`
    the permittivity of the suspended particle (e.g., cells).

    """
    return epsi_med * (
        ((2.0 * epsi_med + epsi_p) - 2.0 * p * (epsi_med - epsi_p))
        / ((2.0 * epsi_med + epsi_p) + p * (epsi_med - epsi_p))
    )


def F(root, epsi_p, epsi_med, p):
    """F factor."""
    return (
        1.0
        / (1.0 - p)
        * (root - epsi_p)
        / (epsi_med - epsi_p)
        * (epsi_med / root) ** (1.0 / 3.0)
    )


def bh_eps_model(epsi_med, epsi_p, p):
    r"""Complex permittvitiy of double shell model by Bruggeman-Hanai approach.

    Parameters
    ----------
    epsi_med: :class:`numpy.ndarray`, complex
        Complex permittivity array of medium
    epsi_p: :class:`numpy.ndarray`, complex
        Complex permittivity array of suspended particles
    p: double
        volume fraction

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Complex permittivity array

    Notes
    -----
    The implementation follows [Cottet2019]_.
    Note that the approach here might not be numerically
    as accurate as the cubic equation solver
    (:meth:`impedancefitter.suspensionmodels.bhcubic_eps_model`).
    In the respective unit test, the deviation is below 1%, though.

    References
    ----------
    .. [Cottet2019] Cottet, J., Fabregue, O., Berger, C., Buret, F., Renaud, P.,
                    & Frénéa-Robin, M. (2019).
                    MyDEP: A New Computational Tool for Dielectric Modeling
                    of Particles and Cells.
                    Biophysical Journal, 116(1), 12–18.
                    https://doi.org/10.1016/j.bpj.2018.11.021

    See Also
    --------
    :meth:`impedancefitter.suspensionmodels.bhcubic_eps_model`

    """
    # initial values
    epsi_sus_n = np.copy(epsi_med)
    # solution
    epsi_sus_n1 = np.zeros(epsi_sus_n.shape, dtype=np.complex128)
    N = 200.0  # use N steps, attention: hard-coded!
    h_p = p / N
    for i in range(int(N) + 1):
        epsi_sus_n1 = epsi_sus_n + h_p / (1.0 - h_p * float(i)) * (
            3.0 * epsi_sus_n * (epsi_p - epsi_sus_n)
        ) / (2.0 * epsi_sus_n + epsi_p)
        epsi_sus_n = epsi_sus_n1
    return epsi_sus_n


def bhcubic_eps_model(epsi_med, epsi_p, p):
    r"""Complex permittvitiy of concentrated suspension
        model by Bruggeman-Hanai approach.

    Parameters
    ----------
    epsi_med: :class:`numpy.ndarray`, complex
        Complex permittivity array of medium
    epsi_p: :class:`numpy.ndarray`, complex
        Complex permittivity array of suspended particles
    p: double
        volume fraction

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Complex permittivity array

    Notes
    -----
    The complex permittivity of the suspension
    :math:`\varepsilon_\mathrm{sus}^\ast` is given by [Hanai1979]_

    .. math::
        \frac{\varepsilon_\mathrm{sus}^\ast -
        \varepsilon_\mathrm{p}^\ast}
        {\varepsilon_\mathrm{med}^\ast - \varepsilon_\mathrm{p}^\ast}
        \left(\frac{\varepsilon_\mathrm{med}^\ast}{\varepsilon_\mathrm{sus}^\ast}\right)^{1/3}
        = 1 - p
        \enspace ,

    with :math:`\varepsilon_\mathrm{med}^\ast` being the permittivity of the
    liquid medium and :math:`\varepsilon_\mathrm{p}^\ast` the permittivity of
    the suspended particle (e.g., cells).

    Cubing the equation yields a cubic equation.
    The cubic roots (three in total) are the possible solutions for
    the complex permittivity
    of the suspension. Only one of them is physical,
    which can be found by substituting
    the cubic roots into another function [Hanai1979]_.

    A numerical solution is implemented in
    :meth:`impedancefitter.suspensionmodels.bh_eps_model`

    References
    ----------
    .. [Hanai1979] Hanai, T., Asami, K., & Korzumi, N. (1979).
                   Dielectric Theory of Concentrated Suspensions
                   of Shell-Spheres in Particular Reference to the
                   Analysis of Biological Cell Suspensions.
                   Bull. Inst. Chem. Res., Kyoto Univ, 57(4), 297–305.
                   http://hdl.handle.net/2433/76842

    See Also
    --------
    :meth:`impedancefitter.suspensionmodels.bh_eps_model`

    """
    epsi_sus = np.zeros(epsi_med.shape, dtype=np.complex128)
    for i in range(len(epsi_sus)):
        # Hanai: https://repository.kulib.kyoto-u.ac.jp/dspace/handle/2433/76842
        r1, r2, r3 = get_cubic_roots(
            -3.0 * epsi_p[i],
            (
                3.0 * np.power(epsi_p[i], 2.0)
                + np.power((p - 1.0) * (epsi_med[i] - epsi_p[i]), 3.0) / epsi_med[i]
            ),
            -np.power(epsi_p[i], 3.0),
        )
        for r in [r1, r2, r3]:
            Fvalue = F(r, epsi_p[i], epsi_med[i], p)
            if np.isclose(Fvalue.real, 1):
                epsi_sus[i] = r
                continue
    return epsi_sus


def eps_sus_ellipsoid_MW(epsi_med, epsi_px, epsi_py, epsi_pz, p, Rx, Ry, Rz):
    r"""Maxwell-Wagner mixture model for dilute suspensions of ellipsoidal particles:w.

    Parameters
    ----------
    epsi_med: :class:`numpy.ndarray`, complex
        complex permittivities of medium
    epsi_px: :class:`numpy.ndarray`, complex
        complex permittivities of suspension phase in x-direction (cells, particles...)
    epsi_py: :class:`numpy.ndarray`, complex
        complex permittivities of suspension phase in y-direction (cells, particles...)
    epsi_pz: :class:`numpy.ndarray`, complex
        complex permittivities of suspension phase in z-direction (cells, particles...)
    p: double
        volume fraction
    Rx: double
        radius for x-semiaxis, value for :math:`R_\mathrm{x}`
    Ry: double
        radius for y-semiaxis, value for :math:`R_\mathrm{y}`
    Rz: double
        radius for z-semiaxis, value for :math:`R_\mathrm{z}`

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Complex permittivity array

    Notes
    -----
    The complex permittivity of the suspension
    :math:`\varepsilon_\mathrm{sus}^\ast` is given by

    .. math::
        \varepsilon_\mathrm{sus}^\ast = \frac{\varepsilon_\mathrm{med}^\ast(1 + 2 K)}
                                        {1-K} \enspace ,

    where

    .. math::
        K = \frac{p}{9} \sum_{k=x,y,z}
        \frac{\varepsilon_\mathrm{p}^\ast - \varepsilon_\mathrm{med}^\ast}
        {\varepsilon_\mathrm{med}^\ast + (\varepsilon_\mathrm{p}^\ast -
        \varepsilon_\mathrm{med}^\ast)L_k}


    with :math:`\varepsilon_\mathrm{med}^\ast` being the permittivity of the
    liquid medium and :math:`\varepsilon_\mathrm{p}^\ast`
    the permittivity of the suspended particle (e.g., cells).
    The depolarization factor :math:`L_k` is implemented in :meth:`Lk`.

    Note that this approximation is valid only for :math:`p \ll 1`.
    An overview of alternative approaches can, for example,
    be found in [Asami2002]_ (e.g., in Table 2 therein).

    References
    ----------
    .. [Asami2002] Asami, K. (2002).
                   Characterization of heterogeneous systems by dielectric spectroscopy.
                   Progress in Polymer Science, 27(8), 1617–1659.
                   https://doi.org/10.1016/S0079-6700(02)00015-1

    """
    if np.any(np.less_equal(np.array([Rx, Ry, Rz]), 1e-3)):
        raise RuntimeError(
            "Attention! For numerical reasons, the radii are evaluated in um! "
            "Please scale the parameter accordingly"
        )

    sum_components = np.zeros(epsi_med.shape, dtype=np.complex128)
    epsi_p = [epsi_px, epsi_py, epsi_pz]
    for i in range(3):
        L_i = Lk(Rx, Ry, Rz, i)
        sum_components += (epsi_p[i] - epsi_med) / (
            epsi_med + (epsi_p[i] - epsi_med) * L_i
        )
    K = p / 9.0 * sum_components
    return epsi_med * (1.0 + 2.0 * K) / (1.0 - K)
