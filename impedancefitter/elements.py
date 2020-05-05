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

import numpy as np
import logging
logger = logging.getLogger('impedancefitter-logger')


def parallel(Z1, Z2):
    """Return values of parallel circuit.

    Parameters
    ----------
    Z1: :class:`numpy.ndarray`, complex or real
        Impedance 1
    Z2: :class:`numpy.ndarray`, complex or real
        Impedance 2

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array
    """
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

        Z_\mathrm{CPE} = k^{-1} (j \omega)^{-\alpha}

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
    return (1. / k) * np.power(1j * omega, -alpha)


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
    return 1. / (1j * omega * C)


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


def Z_stray(omega, C_stray):
    """Stray capacitance in pF

    Parameters
    ----------
    omega: :class:`numpy.ndarray`
        List of frequencies.
    C_stray: double
        Stray capacitance, for numerical reasons in pF.
    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array
    """
    if np.isclose(C_stray, 0, atol=1e-5):
        logger.debug("""Stray capacitance is too small to be added.
                     Did you maybe forget to enter it in terms of pF?""")

    return Z_C(omega, C_stray * 1e-12)


def Z_ws(omega, Aw, B):
    """Warburg short element

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

    .. todo:: Better documentation needed.

    """

    return Aw / np.sqrt(1j * omega) * np.tanh(B * np.sqrt(1j * omega))


def Z_wo(omega, Aw, B):
    """Warburg open element

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

    .. todo:: Better documentation needed.


    """

    return Aw / np.sqrt(1j * omega) / np.tanh(B * np.sqrt(1j * omega))
