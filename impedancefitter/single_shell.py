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


import os
import numpy as np
import matplotlib.pyplot as plt
from .utils import Z_CPE, compare_to_data

if os.path.isfile('./constants.py'):
    import importlib.util
    spec = importlib.util.spec_from_file_location("module.name", os.getcwd() + "/constants.py")
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)

else:
    import impedancefitter.constants as constants

v1 = (1. - constants.dm / constants.Rc)**3


def single_shell_model(omega, k, alpha, em, km, kcp, kmed, emed):
    '''
    formulas for the single shell model are described here
         -returns: Calculated impedance corrected by the constant-phase-element
    '''
    epsi_cp = constants.ecp - 1j * kcp / (constants.e0 * omega)
    epsi_m = em - 1j * km / (constants.e0 * omega)
    epsi_med = emed - 1j * kmed / (constants.e0 * omega)
    # model
    E1 = epsi_cp / epsi_m
    epsi_cell = epsi_m * (2. * (1. - v1) + (1. + 2. * v1) * E1) / ((2. + v1) + (1. - v1) * E1)

    # electrode polarization and calculation of Z
    E0 = epsi_cell / epsi_med
    esus = epsi_med * (2. * (1. - constants.p) + (1. + 2. * constants.p) * E0) / ((2. + constants.p) + (1. - constants.p) * E0)
    Ys = 1j * esus * omega * constants.c0 + 1j * omega * constants.cf                 # cell suspension admittance spectrum
    Zs = 1 / Ys
    Zep = Z_CPE(omega, k, alpha)               # including EP
    Z = Zs + Zep
    return Z


def single_shell_residual(params, omega, data):
    '''
    calculates the residual for the single-shell model, using the single_shell_model.
    if the
    Mode=='Matlab' uses k and e.real as data and data_i
    :param data:
        if data_i is not give, the array in data will be the Impedance calculated from the input file
        if data_i is given, it will be the real part of the permitivity calculated from the input file
    :param data_i:
        if this is given, it will be the conductivity calculated from the input file
    :param params:
        parameters used fot the calculation of the impedance
    '''
    k = params['k'].value
    alpha = params['alpha'].value
    em = params['em'].value
    km = params['km'].value
    kcp = params['kcp'].value
    kmed = params['kmed'].value
    emed = params['emed'].value
    Z_fit = single_shell_model(omega, k, alpha, em, km, kcp, kmed, emed)
    residual = (data - Z_fit) * (data - Z_fit)
    return residual.view(np.float)


def plot_single_shell(omega, Z, result, filename):
    '''
    plot the real part, imaginary part vs frequency and real vs. imaginary part
    '''
    # calculate fitted Z function
    Z_fit = get_single_shell_impedance(omega, result)

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


def get_single_shell_impedance(omega, result):
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
    return single_shell_model(omega, *popt)
