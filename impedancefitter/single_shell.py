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
from .utils import Z_CPE, compare_to_data
from scipy.constants import epsilon_0 as e0


def single_shell_model(omega, constants, k, alpha, em, km, kcp, kmed, emed):
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
    ecp = constants['ecp']
    p = constants['p']
    c0 = constants['c0']
    cf = constants['cf']
    v1 = constants['v1']

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
    Zep = Z_CPE(omega, k, alpha)  # including EP
    Z = Zs + Zep
    return Z


def single_shell_residual(params, omega, data, constants):
    '''
    calculates the residual for the single-shell model, using the single_shell_model.
    '''
    k = params['k'].value
    alpha = params['alpha'].value
    em = params['em'].value
    km = params['km'].value
    kcp = params['kcp'].value
    kmed = params['kmed'].value
    emed = params['emed'].value
    Z_fit = single_shell_model(omega, constants, k, alpha, em, km, kcp, kmed, emed)
    residual = (data - Z_fit)
    return residual.view(np.float)


def plot_single_shell(omega, Z, result, filename, constants):
    '''
    plot the real part, imaginary part vs frequency and real vs. imaginary part
    '''
    # calculate fitted Z function
    Z_fit = get_single_shell_impedance(omega, result, constants)

    # plot real  Impedance part
    plt.figure()
    plt.suptitle("single shell " + str(filename), y=1.05)
    plt.subplot(221)
    plt.xscale('log')
    plt.title("Z_real_part")
    plt.plot(omega, Z_fit.real, '+', label='fitted by Python')
    plt.plot(omega, Z.real, 'r', label='data')
    plt.legend()
    # plot imaginaray Impedance part
    plt.subplot(222)
    plt.title(" Z_imag_part")
    plt.xscale('log')
    plt.plot(omega, Z_fit.imag, '+', label='fitted by Python')
    plt.plot(omega, Z.imag, 'r', label='data')
    plt.legend()
    # plot real vs  imaginary Partr
    plt.subplot(223)
    plt.title("real vs imag")
    plt.plot(Z_fit.real, Z_fit.imag, '+', label="Z_fit")
    plt.plot(Z.real, Z.imag, 'o', label="Z")
    plt.legend()
    compare_to_data(omega, Z, Z_fit, filename, subplot=224)
    plt.tight_layout()
    plt.show()


def get_single_shell_impedance(omega, result, constants):
    # calculate fitted Z function
    popt = np.fromiter([result.params['k'],
                        result.params['alpha'],
                        result.params['em'],
                        result.params['km'],
                        result.params['kcp'],
                        result.params['kmed'],
                        result.params['emed'],
                        ],
                       dtype=np.float)
    return single_shell_model(omega, constants, *popt)
