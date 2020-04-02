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
import matplotlib.pyplot as plt
from .elements import Z_CPE
from .utils import compare_to_data
from scipy.constants import epsilon_0 as e0


def single_shell_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, cf, dm, Rc):
    r"""
    Equations for the single-shell-model( :math:`\nu_1` is calculated like in the double-shell-model):

    .. math::

        \varepsilon_\mathrm{cell}^* = \varepsilon_\mathrm{m}^* * \frac{(2 * (1 - \nu_1) + (1 + 2 * \nu_1) * E_1}{((2 + \nu_1) + (1 - -\nu_1) * E_1}

    .. math::

        E_1 = \frac{\varepsilon_\mathrm{cp}^*}{\varepsilon_\mathrm{m}^*}

    .. math::

        \varepsilon_\mathrm{sus}^* = \varepsilon_\mathrm{med}^* * \frac{2 * (1- p) + (1 + 2 * p) * E_0}{(2 + p) + (1- p) * E_0}

    .. math::

        E_0  = \frac{\varepsilon_\mathrm{cell}^*}{\varepsilon_\mathrm{med}^*}

    """

    v1 = (1. - dm / Rc)**3

    epsi_cp = ecp - 1j * kcp / (e0 * omega)
    epsi_m = em - 1j * km / (e0 * omega)
    epsi_med = emed - 1j * kmed / (e0 * omega)
    # model
    E1 = epsi_cp / epsi_m
    epsi_cell = epsi_m * (2. * (1. - v1) + (1. + 2. * v1) * E1) / ((2. + v1) + (1. - v1) * E1)

    # electrode polarization and calculation of Z
    E0 = epsi_cell / epsi_med
    esus = epsi_med * (2. * (1. - p) + (1. + 2. * p) * E0) / ((2. + p) + (1. - p) * E0)
    Ys = 1j * esus * omega * c0 + 1j * omega * cf  # cell suspension admittance spectrum
    Zs = 1 / Ys
    return Zs


def single_shell_residual(params, omega, data):
    '''
    calculates the residual for the single-shell model, using the single_shell_model.
    '''
    try:
        em = params['em'].value
        km = params['km'].value
        kcp = params['kcp'].value
        kmed = params['kmed'].value
        emed = params['emed'].value
        ecp = params['ecp'].value
        p = params['p']
        c0 = params['c0'] * 1e-12  # use pF as unit
        cf = params['cf'] * 1e-12  # use pF as unit
        dm = params['dm']
        Rc = params['Rc']
    except KeyError as e:
        print(str(e))
        print("You must have forgotten one of the following parameters:\n",
              "em, km, kcp, kmed, emed, ecp, p, c0, cf, Rc")

    Z_fit = single_shell_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, cf, dm, Rc)
    if 'k' in params and 'alpha' in params:
        k = params['k'].value
        alpha = params['alpha'].value
        Z_fit = Z_fit + Z_CPE(omega, k, alpha)  # including EP
    residual = (data - Z_fit)
    return residual.view(np.float)


def plot_single_shell(omega, Z, result, filename):
    '''
    plot the real part, imaginary part vs frequency and real vs. imaginary part
    '''
    # calculate fitted Z function
    Z_fit = get_single_shell_impedance(omega, result.params)

    # plot real  Impedance part
    plt.figure()
    plt.suptitle("single shell " + str(filename), y=1.05)
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
    plt.title("Z imag part")
    plt.ylabel(r"$\Im(Z) [\Omega]$")
    plt.xlabel("frequency [Hz]")
    plt.xscale('log')
    plt.plot(omega / (2. * np.pi), Z_fit.imag, '+', label='fitted')
    plt.plot(omega / (2. * np.pi), Z.imag, 'r', label='data')
    plt.legend()
    # plot real vs  imaginary Partr
    plt.subplot(223)
    plt.title("real vs imag")
    plt.xlabel(r"$\Re(Z) [\Omega]$")
    plt.ylabel(r"$\Im(Z) [\Omega]$")

    plt.plot(Z_fit.real, Z_fit.imag, '+', label="fit")
    plt.plot(Z.real, Z.imag, 'o', label="data")
    plt.legend()
    compare_to_data(omega, Z, Z_fit, filename, subplot=224)
    plt.tight_layout()
    plt.show()


def get_single_shell_impedance(omega, params):
    """
    Provide the angular frequency as well as the result from the fitting procedure.
    The dictionary `params` is processed.
    """
    # calculate fitted Z function
    popt = np.fromiter([params['em'],
                        params['km'],
                        params['kcp'],
                        params['ecp'],
                        params['kmed'],
                        params['emed'],
                        params['p'],
                        params['c0'] * 1e-12,  # use pF as unit
                        params['cf'] * 1e-12,  # use pF as unit
                        params['dm'],
                        params['Rc']
                        ],
                       dtype=np.float)

    Z_s = single_shell_model(omega, *popt)
    if 'k' in params and 'alpha' in params:
        Z_s = Z_s + Z_CPE(omega, params['k'], params['alpha'])
    return Z_s
