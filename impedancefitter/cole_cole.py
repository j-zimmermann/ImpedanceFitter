#    The ImpedanceFitter is a package to fit impedance spectra to equivalent-circuit models using open-source software.
#
#    Copyright (C) 2018, 2019 Leonard Thiele, leonard.thiele[AT]uni-rostock.de
#    Copyright (C) 2018, 2019, 2020 Julius Zimmermann, julius.zimmermann[AT]uni-rostock.de
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

from scipy.constants import epsilon_0 as e0
import numpy as np


def cole_cole_model(omega, c0, epsl, tau, a, sigma, epsinf):
    r"""Cole-Cole model for dielectric properties.

    The model was implemented as presented in [Sabuncu2012]_.
    You need to provide the unit capacitance of your device to get
    the dielectric properties of the Cole-Cole model.

    Parameters
    -----------

    omega: :class:`numpy.ndarray`, double
        list of frequencies
    c0: double
        value for :math:`c_0`, unit capacitance in pF
    epsinf: double
        value for :math:`\varepsilon_\infty`
    epsl: double
        value for :math:`\varepsilon_\mathrm{l}`
    tau: double
        value for :math:`\tau`, in ns
    sigma: double
        value for :math:`\sigma_\mathrm{dc}`
    a: double
        value for :math:`1 - \alpha = a`

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array


    Notes
    -----

    .. warning::

        The unit capacitance is in pF!
        The time constant tau is in ns!

    Equations for calculations:

    .. math::

        \varepsilon^\ast = \varepsilon_\infty + \frac{\varepsilon_\mathrm{l}-\varepsilon_\infty}{1+(j \omega \tau)^a} - \frac{j \sigma_\mathrm{dc}}{\omega \varepsilon_\mathrm{0}}

    .. math::

        Z = \frac{1}{j\varepsilon^\ast \omega c_\mathrm{0}}


    References
    ----------
    .. [Sabuncu2012] Sabuncu, A. C., Zhuang, J., Kolb, J. F., & Beskok, A. (2012).
           Microfluidic impedance spectroscopy as a tool for quantitative biology and biotechnology.
           Biomicrofluidics, 6(3). https://doi.org/10.1063/1.4737121

    """
    tau *= 1e-9  # use ns as unit
    c0 *= 1e-12  # use pF as unit
    es = epsinf + (epsl - epsinf) / (1. + np.power((1j * omega * tau), a)) - 1j * sigma / (omega * e0)

    Z_fit = 1. / (1j * omega * c0 * es)

    return Z_fit


def cole_cole_R_model(omega, Rinf, R0, tau, a):
    r"""Standard Cole-Cole circuit for macroscopic quantities.

    See for example [Schwan1957]_ for more information.


    Parameters
    -----------

    omega: :class:`numpy.ndarray`, double
        list of frequencies
    Rinf: double
        value for :math:`R_\infty`
    R0: double
        value for :math:`R_0`
    tau: double
        value for :math:`\tau`, in ns
    a: double
        value for :math:`1 - \alpha = a`


    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----

    Equation for calculations:


    .. math::

        Z_\mathrm{Cole} = R_\infty + \frac{R_0-R_\infty}{1+(j\omega \tau)^a}

    .. warning::

        The time constant tau is in ns!


    References
    ----------
    .. [Schwan1957] Schwan, H. P. (1957). Electrical properties of tissue and cell suspensions.
           Advances in biological and medical physics (Vol. 5).
           ACADEMIC PRESS INC. https://doi.org/10.1016/b978-1-4832-3111-2.50008-0
    """
    tau *= 1e-9  # use ns as unit
    Z_fit = Rinf + (R0 - Rinf) / (1. + np.power(1j * omega * tau, a))
    return Z_fit


def cole_cole_4_model(omega, c0, epsinf, deps1, deps2, deps3, deps4, tau1, tau2, tau3, tau4, a1, a2, a3, a4, sigma):
    r"""Standard 4-Cole-Cole impedance model.


    Parameters
    -----------

    omega: :class:`numpy.ndarray`, double
        list of frequencies
    c0: double
        value for unit capacitance, in pF
    epsinf: double
        value for :math:`\varepsilon_\infty`
    deps1: double
        value for :math:`\Delta\varepsilon_1`
    deps2: double
        value for :math:`\Delta\varepsilon_2`
    deps3: double
        value for :math:`\Delta\varepsilon_3`
    deps4: double
        value for :math:`\Delta\varepsilon_4`
    tau1: double
        value for :math:`\tau_1`, in ps
    tau2: double
        value for :math:`\tau_2`, in ns
    tau3: double
        value for :math:`\tau_3`, in us
    tau4: double
        value for :math:`\tau_4`, in ms
    a1: double
        value for :math:`1 - \alpha_1 = a`
    a2: double
        value for :math:`1 - \alpha_2 = a`
    a3: double
        value for :math:`1 - \alpha_3 = a`
    a4: double
        value for :math:`1 - \alpha_4 = a`
    sigma: double
        conductivity value

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----
    The original model has been described in [Gabriel1996]_.

    References
    ----------
    .. [Gabriel1996] Gabriel, S., Lau, R. W., & Gabriel, C. (1996).
                    The dielectric properties of biological tissues: III. Parametric models for the dielectric spectrum of tissues.
                    Physics in Medicine and Biology, 41(11), 2271–2293.
                    https://doi.org/10.1088/0031-9155/41/11/003
    """

    c0 *= 1e-12
    tau1 *= 1e-12
    tau2 *= 1e-9
    tau3 *= 1e-6
    tau4 *= 1e-3
    epsc = epsinf - 1j * sigma / (omega * e0)

    epsc += deps1 / (1. + np.power((1j * omega * tau1), a1))
    epsc += deps2 / (1. + np.power((1j * omega * tau2), a2))
    epsc += deps3 / (1. + np.power((1j * omega * tau3), a3))
    epsc += deps4 / (1. + np.power((1j * omega * tau4), a4))

    Z = 1. / (1j * omega * epsc * c0)
    return Z


def cole_cole_3_model(omega, c0, epsinf, deps1, deps2, deps3, tau1, tau2, tau3, a1, a2, a3, sigma):
    r"""Standard 3-Cole-Cole impedance model.


    Parameters
    -----------

    omega: :class:`numpy.ndarray`, double
        list of frequencies
    c0: double
        value for unit capacitance, in pF
    epsinf: double
        value for :math:`\varepsilon_\infty`
    deps1: double
        value for :math:`\Delta\varepsilon_1`
    deps2: double
        value for :math:`\Delta\varepsilon_2`
    deps3: double
        value for :math:`\Delta\varepsilon_3`
    tau1: double
        value for :math:`\tau_1`, in ps
    tau2: double
        value for :math:`\tau_2`, in ns
    tau3: double
        value for :math:`\tau_3`, in us
    a1: double
        value for :math:`1 - \alpha_1 = a`
    a2: double
        value for :math:`1 - \alpha_2 = a`
    a3: double
        value for :math:`1 - \alpha_3 = a`
    sigma: double
        conductivity value

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----
    The original model has been described in [Gabriel1996]_.
    Here, three instead of four dispersions are used.

    """

    c0 *= 1e-12
    tau1 *= 1e-12
    tau2 *= 1e-9
    tau3 *= 1e-6
    epsc = epsinf - 1j * sigma / (omega * e0)

    epsc += deps1 / (1. + np.power((1j * omega * tau1), a1))
    epsc += deps2 / (1. + np.power((1j * omega * tau2), a2))
    epsc += deps3 / (1. + np.power((1j * omega * tau3), a3))

    Z = 1. / (1j * omega * epsc * c0)
    return Z


def cole_cole_2_model(omega, c0, epsinf, deps1, deps2, tau1, tau2, a1, a2, sigma):
    r"""Standard 2-Cole-Cole impedance model.


    Parameters
    -----------

    omega: :class:`numpy.ndarray`, double
        list of frequencies
    c0: double
        value for unit capacitance, in pF
    epsinf: double
        value for :math:`\varepsilon_\infty`
    deps1: double
        value for :math:`\Delta\varepsilon_1`
    deps2: double
        value for :math:`\Delta\varepsilon_2`
    deps3: double
        value for :math:`\Delta\varepsilon_3`
    tau1: double
        value for :math:`\tau_1`, in ps
    tau2: double
        value for :math:`\tau_2`, in ns
    a1: double
        value for :math:`1 - \alpha_1 = a`
    a2: double
        value for :math:`1 - \alpha_2 = a`
    sigma: double
        conductivity value

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----
    The original model has been described in [Gabriel1996]_.
    Here, two instead of four dispersions are used.

    """

    c0 *= 1e-12
    tau1 *= 1e-12
    tau2 *= 1e-9
    epsc = epsinf - 1j * sigma / (omega * e0)

    epsc += deps1 / (1. + np.power((1j * omega * tau1), a1))
    epsc += deps2 / (1. + np.power((1j * omega * tau2), a2))

    Z = 1. / (1j * omega * epsc * c0)
    return Z


def cole_cole_2tissue_model(omega, c0, epsinf, deps1, deps2, tau1, tau2, a1, a2, sigma):
    r"""2-Cole-Cole impedance model for tissues.


    Parameters
    -----------

    omega: :class:`numpy.ndarray`, double
        list of frequencies
    c0: double
        value for unit capacitance, in pF
    epsinf: double
        value for :math:`\varepsilon_\infty`
    deps1: double
        value for :math:`\Delta\varepsilon_1 \cdot 10^3`
    deps2: double
        value for :math:`\Delta\varepsilon_2 \cdot 10^6`
    tau1: double
        value for :math:`\tau_1`, in us
    tau2: double
        value for :math:`\tau_2`, in ms
    a1: double
        value for :math:`1 - \alpha_1 = a`
    a2: double
        value for :math:`1 - \alpha_2 = a`
    sigma: double
        conductivity value

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----
    The original model has been described in [Gabriel1996]_.
    Here, two instead of four dispersions are used.

    """

    c0 *= 1e-12
    tau1 *= 1e-6
    tau2 *= 1e-3
    deps1 *= 1e3
    deps2 *= 1e6
    epsc = epsinf - 1j * sigma / (omega * e0)

    epsc += deps1 / (1. + np.power((1j * omega * tau1), a1))
    epsc += deps2 / (1. + np.power((1j * omega * tau2), a2))

    Z = 1. / (1j * omega * epsc * c0)
    return Z


def havriliak_negami(omega, c0, epsinf, deps, tau, a, beta, sigma):
    r"""Havriliak-Negami relaxation.

    Parameters
    -----------

    omega: :class:`numpy.ndarray`, double
        list of frequencies
    c0: double
        value for :math:`c_0`, unit capacitance in pF
    epsinf: double
        value for :math:`\varepsilon_\infty`
    deps: double
        value for :math:`\Delta\varepsilon`
    tau: double
        value for :math:`\tau`, in ns
    sigma: double
        value for :math:`\sigma_\mathrm{dc}`
    a: double
        value for :math:`1 - \alpha = a`
    beta: double
        value for :math:`\beta`

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array


    Notes
    -----

    .. warning::

        The unit capacitance is in pF!
        The time constant tau is in ns!

    Equations for calculations:

    .. math::

        \varepsilon^\ast = \varepsilon_\infty + \frac{\Delta\varepsilon}{\left(1 + (j \omega \tau)^{a}\right)^\beta} - \frac{j\sigma_{\mathrm{DC}}}{\omega \varepsilon_0} \enspace ,

    .. math::

        Z = \frac{1}{j\varepsilon^\ast \omega c_\mathrm{0}}

    """

    c0 *= 1e-12
    tau *= 1e-9
    epsc = epsinf + deps / np.power(1. + np.power(1j * omega * tau, a), beta) - 1j * sigma / (omega * e0)

    Z = 1. / (1j * omega * epsc * c0)
    return Z


def havriliak_negamitissue(omega, c0, epsinf, deps, tau, a, beta, sigma):
    r"""Havriliak-Negami relaxation.

    Parameters
    -----------

    omega: :class:`numpy.ndarray`, double
        list of frequencies
    c0: double
        value for :math:`c_0`, unit capacitance in pF
    epsinf: double
        value for :math:`\varepsilon_\infty`
    deps: double
        value for :math:`\Delta\varepsilon`
    tau: double
        value for :math:`\tau`, in :math:`\mu`\ s
    sigma: double
        value for :math:`\sigma_\mathrm{dc}`
    a: double
        value for :math:`1 - \alpha = a`
    beta: double
        value for :math:`\beta`

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array


    Notes
    -----

    .. warning::

        The unit capacitance is in pF!
        The time constant tau is in :math:`\mu`\ s!

    Equations for calculations:

    .. math::

        \varepsilon^\ast = \varepsilon_\infty + \frac{\Delta\varepsilon}{\left(1 + (j \omega \tau)^{a}\right)^\beta} - \frac{j\sigma_{\mathrm{DC}}}{\omega \varepsilon_0} \enspace ,

    .. math::

        Z = \frac{1}{j\varepsilon^\ast \omega c_\mathrm{0}}


    """

    c0 *= 1e-12
    tau *= 1e-6
    deps *= 1e3
    epsc = epsinf + deps / np.power(1. + np.power(1j * omega * tau, a), beta) - 1j * sigma / (omega * e0)

    Z = 1. / (1j * omega * epsc * c0)
    return Z
