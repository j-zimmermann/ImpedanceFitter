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

import numpy as np


def log(Z1, dummy):
    """Return logarithm of impedance for LMFIT

    Parameters
    ----------
    Z1: :class:`numpy.ndarray`, complex
        Impedance 1 (model fit)
    dummy : :class:`numpy.ndarray`, complex
        Array full of ones, such that np.log10(dummy) is zero.
    """

    dummy = np.ones(Z1.shape)
    return np.log10(Z1) + np.log10(dummy)


def eps(Z1, make_eps):
    """Return complex permittivity for LMFIT 

    Parameters
    ----------
    Z1: :class:`numpy.ndarray`, complex
        Impedance 1 (model fit)
    make_eps : :class:`numpy.ndarray`, complex
        Output of function :func:`impedancefitter.utils.make_eps`
    """

    return make_eps / Z1


def parallel(Z1, Z2):
    """Return values of parallel circuit.

    Parameters
    ----------
    Z1: :class:`numpy.ndarray`, complex
        Impedance 1
    Z2: :class:`numpy.ndarray`, complex
        Impedance 2

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array
    """
    # to catch infs?
    # return 1. / ((1. / Z1) + (1. / Z2))
    return (Z1 * Z2) / (Z1 + Z2)


def Z_R(omega, R):
    """Create array for a resistor.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`
        List of frequencies.
    R: double
        Resistance.

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    """
    new = np.zeros_like(omega, dtype='complex128')
    return new + R


def Z_L(omega, L):
    """Impedance of an inductor.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`
        List of frequencies.
    L: double
        inductance

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    """
    return 1j * omega * L


def Z_CPE(omega, k, alpha):
    r"""
    CPE impedance

    .. math::

        Z_\mathrm{CPE} = k (j \omega)^{-\alpha}

    Parameters
    ----------
    omega: :class:`numpy.ndarray`
        List of frequencies.
    k: double
        CPE factor
    alpha: double
        CPE phase
    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    """
    Z = k * np.power(1j * omega, -alpha)
    # catch DC case?
    # Z[np.isnan(Z)] = np.inf
    return Z


def Z_C(omega, C):
    """Capacitor impedance

    Parameters
    ----------
    omega: :class:`numpy.ndarray`
        List of frequencies.
    C: double
        capacitance of capacitor
    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array
    """
    Z = 1. / (1j * omega * C)
    # catch DC case?
    # Z[np.isnan(Z)] = np.inf
    return Z


def Z_w(omega, Aw):
    r"""Warburg element

    .. math::

        Z_\mathrm{W} = A_\mathrm{W} \frac{1-j}{\sqrt{\omega}}

    Parameters
    ----------
    omega: :class:`numpy.ndarray`
        List of frequencies.

    Aw: double
        Warburg coefficient

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    """
    return Aw * (1. - 1j) / np.sqrt(omega)


def Z_stray(omega, Cs):
    """Stray capacitance in pF

    Parameters
    ----------
    omega: :class:`numpy.ndarray`
        List of frequencies.
    Cs: double
        Stray capacitance, for numerical reasons in pF.
    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array
    """
    return Z_C(omega, Cs * 1e-12)


def Z_ws(omega, Aw, B):
    r"""Warburg short element

    Parameters
    ----------
    omega: :class:`numpy.ndarray`
        List of frequencies.

    Aw: double
        Warburg coefficient
    B: double
        Second coefficient

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----

    This element is also referred to as finite-length Warburg element
    with transmissive boundary [Barsoukov2018]_.
    Here, the formulation

    .. math::

        Z_\mathrm{W} = A_\mathrm{W} \frac{\tanh\left(B \sqrt{j \omega}\right)}{\sqrt{j \omega}}

    is implemented.

    """

    return Aw / np.sqrt(1j * omega) * np.tanh(B * np.sqrt(1j * omega))


def Z_wo(omega, Aw, B):
    r"""Warburg open element

    Parameters
    ----------
    omega: :class:`numpy.ndarray`
        List of frequencies.

    Aw: double
        Warburg coefficient
    B: double
        Second coefficient

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array


    Notes
    -----

    This element is also referred to as finite-length Warburg element
    with reflective boundary [Barsoukov2018]_.
    Here, the formulation

    .. math::

        Z_\mathrm{W} = \frac{A_\mathrm{W}}{\tanh\left(B \sqrt{j \omega}\right)\sqrt{j \omega}}

    is implemented.

    """

    return Aw / np.sqrt(1j * omega) / np.tanh(B * np.sqrt(1j * omega))
