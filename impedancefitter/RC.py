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

from .elements import Z_in, Z_CPE, Z_loss
from .utils import compare_to_data


def RC_model(omega, cf, Rd, Cd, k=None, alpha=None, L=None, C=None, R=None):
    """
    Simple RC model
    """
    Cfit = Cd + cf
    Zs_fit = Rd / (1. + 1j * omega * Cfit * Rd)
    if k is not None and alpha is not None:
        Zep_fit = Z_CPE(omega, k, alpha)
        Zs_fit = Zs_fit + Zep_fit
    if L is not None:
        if C is None:
            Zin_fit = Z_in(omega, L, R)
        elif C is not None and R is not None:
            Zin_fit = Z_loss(omega, L, C, R)
        Zs_fit = Zs_fit + Zin_fit
    return Zs_fit


def RC_residual(params, omega, data):
    """
    use the plain suspension model and calculate the residual (the difference between data and fitted values)
    """
    Rd = params['Rd'].value
    Cd = params['Cd'].value
    cf = params['cf'].value * 1e-12  # use pF as unit
    alpha = None
    k = None
    L = None
    C = None
    R = None
    if 'alpha' in params:
        alpha = params['alpha'].value
    if 'k' in params:
        k = params['k'].value
    if 'L' in params:
        L = params['L'].value * 1e-9  # use nH as units
    if 'C' in params:
        C = params['C'].value * 1e-12  # use pF as units
    if 'R' in params:
        R = params['R'].value
    Z_fit = RC_model(omega, cf, Rd, Cd, k=k, alpha=alpha, L=L, C=C, R=R)
    residual = (data - Z_fit)
    return residual.view(np.float)


def plot_RC(omega, Z, result, filename):
    """
    plot results of cole-cole model and compare fit to data.
    """
    Z_fit = get_RC_impedance(omega, result.params)
    plt.figure()
    plt.suptitle("RC fit plot\n" + str(filename), y=1.05)
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


def get_RC_impedance(omega, params):
    """
    Provide the angular frequency as well as the result from the fitting procedure.
    The dictionary `params` is processed.

    Attention: `c0` and `cf` have to be given in pF!!!
    """
    # calculate fitted Z function
    popt = np.fromiter([params['cf'] * 1e-12,  # use pF as unit
                       params['Rd'],
                       params['Cd']],
                       dtype=np.float)
    kwargs = {}
    if 'k' in params and 'alpha' in params:
        kwargs['k'] = params['k']
        kwargs['alpha'] = params['alpha']
    if 'L' in params:
        kwargs['L'] = params['L'] * 1e-9
    if 'C' in params:
        kwargs['C'] = params['C'] * 1e-12
    if 'R' in params:
        kwargs['R'] = params['R']
    Z_s = RC_model(omega, *popt, **kwargs)
    return Z_s
