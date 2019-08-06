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
import os
from .utils import compare_to_data, Z_CPE


if os.path.isfile('./constants.py'):
    print("Using constants specified in directory.")
    import importlib.util
    spec = importlib.util.spec_from_file_location("module.name", os.getcwd() + "/constants.py")
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)

else:
    print("Using default constants.")
    import impedancefitter.constants as constants


def double_shell_model(omega, k, alpha, km, em, kcp, ene, kne, knp, kmed, emed):
    """
    atm, this is a dummy for the Double-Shell-Model equations,
    containing everything from double_shell_sopt_pso.py to fit the variable definitions in this file
    """
    epsi_m = em + km / (1j * omega * constants.e0)
    epsi_cp = constants.ecp + kcp / (1j * omega * constants.e0)
    epsi_ne = ene + kne / (1j * omega * constants.e0)
    epsi_np = constants.enp + knp / (1j * omega * constants.e0)
    epsi_med = emed + kmed / (1j * omega * constants.e0)

    E3 = epsi_np / epsi_ne
    E2 = ((epsi_ne / epsi_cp) * (2. * (1. - constants.v3) + (1. + 2. * constants.v3) * E3) /
          ((2. + constants.v3) + (1. - constants.v3) * E3))  # Eq. 13
    E1 = ((epsi_cp / epsi_m) * (2. * (1. - constants.v2) + (1. + 2. * constants.v2) * E2) /
          ((2. + constants.v2) + (1. - constants.v2) * E2))  # Eq. 14

    epsi_cell = (epsi_m * (2. * (1. - constants.v1) + (1. + 2. * constants.v1) * E1) /
                 ((2. + constants.v1) + (1. - constants.v1) * E1))  # Eq. 11
    E0 = epsi_cell / epsi_med
    esus = epsi_med * (2. * (1. - constants.p) + (1. + 2. * constants.p) * E0) / ((2. + constants.p) + (1. - constants.p) * E0)
    Ys = 1j * esus * omega * constants.c0 + 1j * omega * constants.cf                 # cell suspension admittance spectrum
    Zs = 1 / Ys
    Zep = Z_CPE(omega, k, alpha)               # including EP
    Z = Zs + Zep
    return Z


def double_shell_residual(params, omega, data):
    '''
    data is Z
    '''
    k = params['k'].value
    alpha = params['alpha'].value
    km = params['km'].value
    em = params['em'].value
    kcp = params['kcp'].value
    ene = params['ene'].value
    kne = params['kne'].value
    knp = params['knp'].value
    kmed = params['kmed'].value
    emed = params['emed'].value

    Z_fit = double_shell_model(omega, k, alpha, km, em, kcp, ene, kne, knp, kmed, emed)
    # define the objective function
    # optimize for impedance
    residual = data - Z_fit
    return residual.view(np.float)


def plot_double_shell(omega, Z, result, filename):
    '''
    plot the real and imaginary part of the impedance vs. the frequency and
    real vs. imaginary part
    '''
    popt = np.fromiter([result.params['k'],
                        result.params['alpha'],
                        result.params['km'],
                        result.params['em'],
                        result.params['kcp'],
                        result.params['ene'],
                        result.params['kne'],
                        result.params['knp'],
                        result.params['kmed'],
                        result.params['emed'],
                        ],
                       dtype=np.float)
    Z_fit = double_shell_model(omega, *popt)

    # plot real  Impedance part
    plt.figure()
    plt.suptitle("double shell " + str(filename), y=1.05)
    plt.xscale('log')
    plt.subplot(221)
    plt.title("Z_real_part")
    plt.plot(omega, Z_fit.real, '+', label='fitted by Python')
    plt.plot(omega, Z.real, 'r', label='data')
    plt.legend()
    # plot imaginaray Impedance part
    plt.subplot(222)
    plt.title("Z_imag_part")
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
