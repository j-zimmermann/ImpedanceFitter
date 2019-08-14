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

from .utils import Z_CPE, e_sus, Z_sus, compare_to_data


def suspension_model(omega, el, tau, a, kdc, eh):
    es = e_sus(omega, eh, el, tau, a)
    Zs_fit = Z_sus(omega, es, kdc)
    Z_fit = Zs_fit
    return Z_fit


def suspension_residual(params, omega, data):
    el = params['epsi_l'].value
    tau = params['tau'].value
    a = params['a'].value
    kdc = params['conductivity'].value
    eh = params['eh'].value
    Z_fit = suspension_model(omega, el, tau, a, kdc, eh)
    residual = (data - Z_fit) * (data - Z_fit)
    return residual.view(np.float)


def cole_cole_model(omega, k, el, tau, a, alpha, kdc, eh):
    """
    function holding the cole_cole_model equations, returning the calculated impedance
    """
    Zep_fit = Z_CPE(omega, k, alpha)
    es = e_sus(omega, eh, el, tau, a)

    Zs_fit = Z_sus(omega, es, kdc)
    Z_fit = Zep_fit + Zs_fit

    return Z_fit


def cole_cole_residual(params, omega, data):
    """
    We have two ways to compute the residual. One as in the Matlab script (data_i is not None) and just a plain fit.
    In the matlab case, the input parameters have to be epsilon.real and k, which are determined in readin_Data_from_file.
    """
    k = params['k'].value
    el = params['epsi_l'].value
    tau = params['tau'].value
    a = params['a'].value
    alpha = params['alpha'].value
    kdc = params['conductivity'].value
    eh = params['eh'].value
    Z_fit = cole_cole_model(omega, k, el, tau, a, alpha, kdc, eh)
    residual = (data - Z_fit) * (data - Z_fit)
    return residual.view(np.float)


def plot_cole_cole(omega, Z, result, filename):
    popt = np.fromiter(result.params.valuesdict().values(), dtype=np.float)
    Z_fit = cole_cole_model(omega, *popt)

    plt.figure()
    plt.suptitle("Cole-Cole fit plot\n" + str(filename), y=1.05)
    # plot real  Impedance part
    plt.subplot(221)
    plt.xscale('log')
    plt.title("Z_real_part")
    plt.plot(omega, Z_fit.real, '+', label='fitted by Python')
    plt.plot(omega, Z.real, 'r', label='data')
    plt.legend()
    # plot imaginaray Impedance part
    plt.subplot(222)
    plt.title(" Z_imaginary_part")
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
