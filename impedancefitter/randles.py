#    The ImpedanceFitter is a package that provides means to fit impedance spectra to theoretical models using open-source software.
#
#    Copyright (C) 2018, 2019 Leonard Thiele, leonard.thiele[AT]uni-rostock.de
#    Copyright (C) 2018, 2019 Julius Zimmermann, julius.zimmermann[AT]uni-rostock.de
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
from .utils import add_additions
from .elements import Z_CPE
import logging

logger = logging.getLogger('impedancefitter-logger')


def Z_randles(omega, R0, Rs, Aw, C0, k=None, alpha=None, L=None, C=None, R=None, cf=None):
    r"""
    function holding the Randles equation with capacitor in parallel to resistor in series with Warburg element
    and another resistor in series.
    It returns the calculated impedance
    Equations for calculations:

    impedance of resistor and Warburg element:

    .. math::

        Z_\mathrm{RW} = R_0 + A_\mathrm{W} \frac{1 - j}{\sqrt{\omega}}

    impedance of capacitor:

    .. math::

        Z_C = (j \omega C_0)^{-1}

    .. math::

        Z_\mathrm{fit} = R_s + \frac{Z_C Z_\mathrm{RW}}{Z_C + Z_\mathrm{RW}}

    """
    Z_RW = R0 + Aw * (1. - 1j) / np.sqrt(omega)
    Z_C = 1. / (1j * omega * C0)
    Z_par = Z_RW * Z_C / (Z_RW + Z_C)
    Zs_fit = Rs + Z_par
    if k is not None or alpha is not None:
        logger.warning("""This model cannot be combined with an electrode polarization correction.
                          The parameters k and alpha will have no effect here.""")
    Z_fit = add_additions(omega, Zs_fit, None, None, L, C, R, cf)
    return Z_fit


def Z_randles_CPE(omega, R0, Rs, Aw, k, alpha, L=None, C=None, R=None, cf=None):
    Z_RW = R0 + Aw * (1. - 1j) / np.sqrt(omega)
    Z_cpe = Z_CPE(omega, k, alpha)
    Z_par = Z_RW * Z_cpe / (Z_RW + Z_cpe)
    Zs_fit = Rs + Z_par
    Z_fit = add_additions(omega, Zs_fit, None, None, L, C, R, cf)
    return Z_fit
