#    The ImpedanceFitter is a package that provides means to fit impedance spectra to theoretical models using open-source software.
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


def _Z_sus(omega, es, kdc, c0):
    r"""Impedance of suspension.

    Described for example in [Sabuncu2012]_.

    Parameters
    -----------

    omega: double or array of double
        list of frequencies
    es: complex
        complex valued permittivity, check e.g. :func:e_sus
    kdc: double
        conductivity
    c0: double
        unit capacitance

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    References
    ----------
    .. [Sabuncu2012] Sabuncu, A. C., Zhuang, J., Kolb, J. F., & Beskok, A. (2012).
           Microfluidic impedance spectroscopy as a tool for quantitative biology and biotechnology.
           Biomicrofluidics, 6(3). https://doi.org/10.1063/1.4737121
    """

    return 1. / (1j * es * omega * c0 + (kdc * c0 / e0))


def _e_sus(omega, eh, el, tau, a):
    r"""
    Complex permitivity after Cole-Cole.
    See the original paper of the Cole brothers [Cole1941]_.
    Difference: the exponent :math:`1 - \alpha` is here named `a`.

    Parameters
    -----------

    omega: :class:`numpy.ndarray`, double
        list of frequencies
    eh: double
        value for :math:`\varepsilon_\infty`
    el: double
        value for :math:`\varepsilon_0`
    tau: double
        value for :math:`\tau_0`
    a: double
        value for :math:`1 - \alpha`

    References
    ----------

    [Cole1941]_ Cole, K. S., & Cole, R. H. (1941). Dispersion and absorption in dielectrics I.
         Alternating current characteristics. The Journal of Chemical Physics,
         9(4), 341–351. https://doi.org/10.1063/1.1750906
    """
    return eh + (el - eh) / (1. + np.power((1j * omega * tau), a))


def cole_cole_model(omega, c0, el, tau, a, kdc, eh):
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
    eh: double
        value for :math:`\varepsilon_\mathrm{h}`
    el: double
        value for :math:`\varepsilon_\mathrm{l}`
    tau: double
        value for :math:`\tau`, in ns
    kdc: double
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

        \varepsilon_\mathrm{s} = \varepsilon_\mathrm{h} + \frac{\varepsilon_\mathrm{l}-\varepsilon_\mathrm{h}}{1+(j \omega \tau)^a}

    .. math::

        Z_\mathrm{s} = \frac{1}{j\varepsilon_\mathrm{s}\omega c_\mathrm{0} + \frac{\sigma_\mathrm{dc} c_\mathrm{0}}{\varepsilon_\mathrm{0}}}



    References
    ----------
    .. [Sabuncu2012] Sabuncu, A. C., Zhuang, J., Kolb, J. F., & Beskok, A. (2012).
           Microfluidic impedance spectroscopy as a tool for quantitative biology and biotechnology.
           Biomicrofluidics, 6(3). https://doi.org/10.1063/1.4737121

    """
    tau *= 1e-9  # use ns as unit
    c0 *= 1e-12  # use pF as unit
    es = _e_sus(omega, eh, el, tau, a)
    Z_fit = _Z_sus(omega, es, kdc, c0)

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


    References
    ----------
    .. [Schwan1957] Schwan, H. P. (1957). Electrical properties of tissue and cell suspensions.
           Advances in biological and medical physics (Vol. 5).
           ACADEMIC PRESS INC. https://doi.org/10.1016/b978-1-4832-3111-2.50008-0
    """
    tau *= 1e-9  # use ns as unit
    Z_fit = Rinf + (R0 - Rinf) / (1. + np.power(1j * omega * tau, a))
    return Z_fit
