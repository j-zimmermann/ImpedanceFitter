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

from .utils import compare_to_data
from .elements import Z_CPE, Z_loss, Z_in


def cole_cole_R_model(omega, cf, Rinf, R0, tau, a, k=None, alpha=None, L=None, C=None, R=None):
    r"""
    function holding the cole_cole_model equations, returning the calculated impedance
    Equations for calculations:

    .. math::

         Z_\mathrm{ep} = k^{-1} \* j\*\omega^{-\alpha}

    .. math::

        \Z_\mathrm{Cole} = R_\infty + \frac{R_0-R_\infty}{1+(j\*\omega\*\tau)^a}

    .. math::

        Z_\mathrm{fit} = Z_\mathrm{Cole} + Z_\mathrm{ep}

    """
    Zs_fit = Rinf + (R0 - Rinf) / (1. + 1j * omega * tau)**a
    # add influence of stray capacitance if it is greater than 0
    if not np.isclose(cf, 0.0):
        # attention!! use pf as units!
        cf *= 1e-12
        Zs_fit += 1. / (1j * omega * cf)
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


def cole_cole_R_residual(params, omega, data):
    """
    compute difference between data and model.
    Attention: `c0` and `cf` in terms of pF, `tau` in ps
    """
    Rinf = params['Rinf'].value
    tau = params['tau'].value * 1e-12  # use ps as unit
    a = params['a'].value
    R0 = params['R0'].value
    cf = params['cf'].value
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
        L = params['L'].value
    if 'C' in params:
        C = params['C'].value
    if 'R' in params:
        R = params['R'].value
    Z_fit = cole_cole_R_model(omega, cf, Rinf, R0, tau, a, k=k, alpha=alpha, L=L, C=C, R=R)
    residual = (data - Z_fit)
    return residual.view(np.float)


def plot_cole_cole_R(omega, Z, result, filename):
    """
    plot results of cole-cole model and compare fit to data.
    """
    Z_fit = get_cole_cole_R_impedance(omega, result.params)
    plt.figure()
    plt.suptitle("Cole-Cole fit plot\n" + str(filename), y=1.05)
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


def get_cole_cole_R_impedance(omega, params):
    """
    Provide the angular frequency as well as the resulting parameters from the fitting procedure.
    The dictionary `params` is processed.
    """
    # calculate fitted Z function
    popt = np.fromiter([params['cf'],
                        params['Rinf'],
                        params['R0'],
                        params['tau'] * 1e-12,  # use ps as unit
                        params['a']],
                       dtype=np.float)
    kwargs = {}
    if 'k' in params and 'alpha' in params:
        kwargs['k'] = params['k']
        kwargs['alpha'] = params['alpha']
    if 'L' in params:
        kwargs['L'] = params['L']
    if 'C' in params:
        kwargs['C'] = params['C']
    if 'R' in params:
        kwargs['R'] = params['R']
    Z_s = cole_cole_R_model(omega, *popt, **kwargs)
    return Z_s
