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


import matplotlib.pyplot as plt
import numpy as np

from .elements import Z_CPE, Z_in, Z_loss
from .utils import compare_to_data

def Z_randles(omega, R0, Rs, Aw, C0,  L=None, C=None, R=None):
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
    Z_par = Z_RW  * Z_C / (Z_RW +  Z_C)
    Zs_fit = Rs + Z_par 
        Zs_fit = Zs_fit + Zep_fit
    if L is not None:
        if C is None:
            Zin_fit = Z_in(omega, L, R)
        elif C is not None and R is not None:
            Zin_fit = Z_loss(omega, L, C, R)
        Zs_fit = Zs_fit + Zin_fit
    return Zs_fit


def Z_randles_CPE(omega, R0, Rs, Aw, k, alpha, L=None, C=None, R=None):
    Z_RW = R0 + Aw * (1. - 1j) / np.sqrt(omega)
    Z_CPE = Z_CPE(omega, k, alpha)
    Z_par = Z_RW  * Z_CPE / (Z_RW +  Z_CPE)
    Zs_fit = Rs + Z_par 
    if L is not None:
        if C is None:
            Zin_fit = Z_in(omega, L, R)
        elif C is not None and R is not None:
            Zin_fit = Z_loss(omega, L, C, R)
        Zs_fit = Zs_fit + Zin_fit
    return Zs_fit


def randles_residual(params, omega, data, cpe=False):
    """
    compute difference between data and model.
    """
    R0 = params['R0'].value
    Rs = params['Rs'].value
    Aw = params['Aw'].value
    
    L = None
    C = None
    R = None
    if 'L' in params:
        L = params['L'].value
    if 'C' in params:
        C = params['C'].value
    if 'R' in params:
        R = params['R'].value

    if cpe:
        alpha = params['alpha'].value
        k = params['k'].value
        Z_fit = Z_randles_CPE(omega, R0, Rs, Aw, k, alpha, L=L, C=C, R=R)
    else:
        C0 = params['C0'].value
        Z_fit = Z_randles(omega, R0, Rs, Aw, C0, L=L, C=C, R=R)

    residual = (data - Z_fit)
    return residual.view(np.float)

def plot_randles(omega, Z, result, filename, cpe=False):
    """
    plot results of randles model and compare fit to data.
    """
    Z_fit = get_randles_impedance(omega, result.params, cpe=cpe)
    plt.figure()
    plt.suptitle("Randles fit plot\n" + str(filename), y=1.05)
    # plot real  Impedance part
    plt.subplot(221)
    plt.xscale('log')
    plt.title("Z real part")
    plt.ylabel(r"$\Re(Z) [\Omega]$")
    plt.xlabel("frequency [Hz]")
    plt.plot(omega / (2. * np.pi), Z_fit.real, '+', label='fitted')
    plt.plot(omega / (2. * np.pi), Z.real, 'r', label='data')
    plt.legend()
    # plot imaginaray Impedance part
    plt.subplot(222)
    plt.title("Z imaginary part")
    plt.xscale('log')
    plt.ylabel(r"$\Im(Z) [\Omega]$")
    plt.xlabel("frequency [Hz]")
    plt.plot(omega / (2. * np.pi), Z_fit.imag, '+', label='fitted')
    plt.plot(omega / (2. * np.pi), Z.imag, 'r', label='data')
    plt.legend()
    # plot real vs  imaginary Part
    plt.subplot(223)
    plt.title("real vs imag")
    plt.ylabel(r"$\Im(Z) [\Omega]$")
    plt.xlabel(r"$\Re(Z) [\Omega]$")
    plt.plot(Z_fit.real, Z_fit.imag, '+', label="fit")
    plt.plot(Z.real, Z.imag, 'o', label="data")
    plt.legend()
    compare_to_data(omega, Z, Z_fit, filename, subplot=224)
    plt.tight_layout()
    plt.show()


def get_randles_impedance(omega, params, cpe=False):
    """
    Provide the angular frequency as well as the resulting parameters from the fitting procedure.
    The dictionary `params` is processed.
    """
    # calculate fitted Z function
    tmp = np.fromiter([params['R0'], 
                       params['Rs'],
                       params['Aw']],
                       dtype=np.float)
    if cpe:
        popt = np.append(tmp, params['k'], params['alpha'])
    else:
        popt = np.append(tmp, params['C0'])


    kwargs = {}
    if 'L' in params:
        kwargs['L'] = params['L']
    if 'C' in params:
        kwargs['C'] = params['C']
    if 'R' in params:
        kwargs['R'] = params['R']

    if cpe:
        Z_s = Z_randles_CPE(omega, *popt, **kwargs)
    else
        Z_s = Z_randles(omega, *popt, **kwargs)
    return Z_s

