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


def Z_loss(omega, L, C, R):
    """
    impedance for high loss materials, where LCR are in parallel.
    Described for instance in 10.1002/cmr.b.21318.

    Parameters
    -----------
    omega: double or array of double
        list of frequencies
    L: double
        inductance of coil
    C: double
        capacitance of capacitor
    R: double
        resistance of resistor
    """
    Y = 1. / R + 1. / (1j * omega * L) + 1j * omega * C
    Z = 1. / Y
    return Z


def Z_in(omega, L, R):
    """
    Lead inductance of wires connecting DUT.
    Described for instance in 10.1002/cmr.b.21318

    Parameters
    -----------
    omega: double or array of double
        list of frequencies
    L: double
        inductance of coil
    R: double
        resistance of resistor

    """
    return R + 1j * omega * L


def Z_CPE(omega, k, alpha):
    r"""
    CPE impedance

    .. math::

        Z_\mathrm{CPE} = k^{-1} (j \omega)^{-\alpha}

    Parameters
    -----------

    omega: double or array of double
        list of frequencies
    k: double
        CPE factor
    alpha: double
        CPE phase

    """
    return (1. / k) * np.power(1j * omega, -alpha)


def Z_C(omega, C):
    """
    capacitor impedance

    Parameters
    -----------

    omega: double or array of double
        list of frequencies
    C: double
        capacitance of capacitor

    """
    return 1. / (1j * omega * C)


def Z_sus(omega, es, kdc, c0):
    r"""
    impedance of suspension as used in paper with DOI: 10.1063/1.4737121

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
    """

    return 1. / (1j * es * omega * c0 + (kdc * c0) / e0)


def e_sus(omega, eh, el, tau, a):
    r"""
    Complex permitivity after Cole-Cole.
    See their paper with DOI: 10.1063/1.1750906
    Difference: the exponent :math:`1 - \alpha` is here named `a`.

    Parameters
    -----------

    omega: double or array of double
        list of frequencies
    eh: double
        value for :math:`\varepsilon_\infty`
    el: double
        value for :math:`\varepsilon_0`
    tau: double
        value for :math:`\tau_0`
    a: double
        value for :math:`1 - \alpha`
    """
    return eh + (el - eh) / (1. + np.power((1j * omega * tau), a))


def Z_w(omega, Aw):
    r"""
    Warburg element

    .. math::

        Z_\mathrm{W} = A_\mathrm{W} \frac{1-j}{\sqrt{\omega}}


    Parameters
    -----------

    omega: double or array of double
        list of frequencies
    A_w: double
        Warburg coefficient
    """
    return Aw * (1. - 1j) / np.sqrt(omega)
