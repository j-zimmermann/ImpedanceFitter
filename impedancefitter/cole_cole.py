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

from .utils import Z_in, Z_CPE, e_sus, Z_sus, compare_to_data, Z_loss


def suspension_model(omega, c0, cf, el, tau, a, kdc, eh):
    """
    Simple suspension model using :func:`impedancefitter.utils.Z_sus`
    """
    es = e_sus(omega, eh, el, tau, a)
    Zs_fit = Z_sus(omega, es, kdc, c0, cf)
    Z_fit = Zs_fit
    return Z_fit


def suspension_residual(params, omega, data):
    """
    use the plain suspension model and calculate the residual (the difference between data and fitted values)
    """
    el = params['epsi_l'].value
    tau = params['tau'].value
    a = params['a'].value
    kdc = params['conductivity'].value
    eh = params['eh'].value
    c0 = params['c0'].value
    cf = params['cf'].value
    Z_fit = suspension_model(omega, c0, cf, el, tau, a, kdc, eh)
    residual = (data - Z_fit) / data
    return residual.view(np.float)


def cole_cole_model(omega, c0, cf, el, tau, a, kdc, eh, k=None, alpha=None, L=None, C=None, R=None):
    r"""
    function holding the cole_cole_model equations, returning the calculated impedance
    Equations for calculations:

    .. math::

         Z_\mathrm{ep} = k^{-1} \* j\*\omega^{-\alpha}

    .. math::

        \varepsilon_\mathrm{s} = \varepsilon_\mathrm{h} + \frac{\varepsilon_\mathrm{l}-\varepsilon_\mathrm{h}}{1+(j\*\omega\*\tau)^a}

    .. math::

        Z_\mathrm{s} = \frac{1}{j\*\varepsilon_\mathrm{s}\*\omega\*c_\mathrm{0} + \frac{\sigma_\mathrm{dc}\*c_\mathrm{0}}{\varepsilon_\mathrm{0}} + j\*\omega\*c_\mathrm{f}}

    .. math::

        Z_\mathrm{fit} = Z_\mathrm{s} + Z_\mathrm{ep}

    find more details in formula 6 and 7 of https://ieeexplore.ieee.org/document/6191683

    """
    es = e_sus(omega, eh, el, tau, a)
    Zs_fit = Z_sus(omega, es, kdc, c0, cf)
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


def cole_cole_residual(params, omega, data):
    """
    compute difference between data and model.
    """
    el = params['epsi_l'].value
    tau = params['tau'].value
    a = params['a'].value
    kdc = params['conductivity'].value
    eh = params['eh'].value
    c0 = params['c0'].value
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
    Z_fit = cole_cole_model(omega, c0, cf, el, tau, a, kdc, eh, k=k, alpha=alpha, L=L, C=C, R=R)
    residual = (data - Z_fit) / data
    return residual.view(np.float)


def plot_cole_cole(omega, Z, result, filename):
    """
    plot results of cole-cole model and compare fit to data.
    """
    Z_fit = get_cole_cole_impedance(omega, result)
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


def get_cole_cole_impedance(omega, result):
    """
    Provide the angular frequency as well as the result from the fitting procedure.
    The result object contains a dictionary `params` that is processed.
    """
    # calculate fitted Z function
    popt = np.fromiter([result.params['c0'],
                       result.params['cf'],
                       result.params['epsi_l'],
                       result.params['tau'],
                       result.params['a'],
                       result.params['conductivity'],
                       result.params['eh']],
                       dtype=np.float)
    kwargs = {}
    if 'k' in result.params and 'alpha' in result.params:
        kwargs['k'] = result.params['k']
        kwargs['alpha'] = result.params['alpha']
    if 'L' in result.params:
        kwargs['L'] = result.params['L']
    if 'C' in result.params:
        kwargs['C'] = result.params['C']
    if 'R' in result.params:
        kwargs['R'] = result.params['R']
    Z_s = cole_cole_model(omega, *popt, **kwargs)
    return Z_s
