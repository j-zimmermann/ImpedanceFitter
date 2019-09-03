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
from scipy.constants import epsilon_0 as e0
from .utils import compare_to_data, Z_CPE


def double_shell_model(omega, constants, km, em, kcp, ene, kne, knp, kmed, emed):
    r"""
    Equations for the double-shell-model:

    .. math::

        \varepsilon_\mathrm{mix}^\ast = \varepsilon_\mathrm{sup}^\ast\frac{(2\varepsilon_\mathrm{sup}^\ast+\varepsilon_\mathrm{c}^\ast)-2p(\varepsilon_\mathrm{sup}^\ast-\varepsilon_\mathrm{c}^\ast)}{(2\varepsilon_\mathrm{sup}^\ast+\varepsilon_\mathrm{c}^\ast)+p(\varepsilon_\mathrm{sup}^\ast-\varepsilon_\mathrm{c}^\ast)}

    .. math::

            \varepsilon_\mathrm{c}^\ast = \varepsilon_\mathrm{m}^\ast\frac{2(1-\nu_\mathrm{1})+(1+2\nu_\mathrm{1})E_\mathrm{1}}{(2+\nu_\mathrm{1})+(1-\nu_\mathrm{1})E_\mathrm{1}}

    .. math::

            E_\mathrm{1}  = \frac{\varepsilon_\mathrm{cp}^\ast}{\varepsilon_\mathrm{m}^\ast} \frac{2(1-\nu_\mathrm{2})+(1+2\nu_\mathrm{2})E_\mathrm{2}}{(2+\nu_\mathrm{2})+(1-\nu_\mathrm{2})E_\mathrm{2}}

    .. math::

            E_\mathrm{2} = \frac{\varepsilon_\mathrm{ne}^\ast}{\varepsilon_\mathrm{cp}^\ast}\frac{2(1-\nu_\mathrm{3})+(1+2\nu_\mathrm{3})E_\mathrm{3}}{(2+\nu_\mathrm{3})+(1-\nu_\mathrm{3})E_\mathrm{3}}

    .. math::

            E_\mathrm{3} = \frac{\varepsilon_\mathrm{np}^\ast}{\varepsilon_\mathrm{ne}^\ast}

    with :math:`R \hat{=}` outer cell Radius; :math:`R_\mathrm{n} \hat{=}` outer Radius of the nucleus; :math:`d \hat{=}` thickness of the membrane

    .. math::

            \nu_\mathrm{1} = \left(1-\frac{d}{R}\right)^3

    .. math::

            \nu_\mathrm{2} = \left(\frac{R_\mathrm{n}}{R-d}\right)^3

    .. math::

            \nu_\mathrm{3} = \left(1-\frac{d_\mathrm{n}}{R_\mathrm{n}}\right)^3

    """
    ecp = constants['ecp']
    enp = constants['enp']
    p = constants['p']
    c0 = constants['c0']
    cf = constants['cf']
    v1 = constants['v1']
    v2 = constants['v2']
    v3 = constants['v3']

    epsi_m = em + km / (1j * omega * e0)
    epsi_cp = ecp + kcp / (1j * omega * e0)
    epsi_ne = ene + kne / (1j * omega * e0)
    epsi_np = enp + knp / (1j * omega * e0)
    epsi_med = emed + kmed / (1j * omega * e0)

    E3 = epsi_np / epsi_ne
    E2 = ((epsi_ne / epsi_cp) * (2. * (1. - v3) + (1. + 2. * v3) * E3) /
          ((2. + v3) + (1. - v3) * E3))  # Eq. 13
    E1 = ((epsi_cp / epsi_m) * (2. * (1. - v2) + (1. + 2. * v2) * E2) /
          ((2. + v2) + (1. - v2) * E2))  # Eq. 14

    epsi_cell = (epsi_m * (2. * (1. - v1) + (1. + 2. * v1) * E1) /
                 ((2. + v1) + (1. - v1) * E1))  # Eq. 11
    E0 = epsi_cell / epsi_med
    esus = epsi_med * (2. * (1. - p) + (1. + 2. * p) * E0) / ((2. + p) + (1. - p) * E0)
    Ys = 1j * esus * omega * c0 + 1j * omega * cf  # cell suspension admittance spectrum
    Zs = 1 / Ys
    return Zs


def double_shell_residual(params, omega, data, constants):
    '''
    data is Z
    '''
    km = params['km'].value
    em = params['em'].value
    kcp = params['kcp'].value
    ene = params['ene'].value
    kne = params['kne'].value
    knp = params['knp'].value
    kmed = params['kmed'].value
    emed = params['emed'].value

    Z_fit = double_shell_model(omega, constants, km, em, kcp, ene, kne, knp, kmed, emed)
    if 'k' in params and 'alpha' in params:
        k = params['k'].value
        alpha = params['alpha'].value
        Z_fit = Z_fit + Z_CPE(omega, k, alpha)  # including EP

    # define the objective function
    # optimize for impedance
    residual = (data - Z_fit)
    return residual.view(np.float)


def plot_double_shell(omega, Z, result, filename, constants):
    '''
    plot the real and imaginary part of the impedance vs. the frequency and
    real vs. imaginary part
    '''
    popt = np.fromiter([result.params['km'],
                        result.params['em'],
                        result.params['kcp'],
                        result.params['ene'],
                        result.params['kne'],
                        result.params['knp'],
                        result.params['kmed'],
                        result.params['emed'],
                        ],
                       dtype=np.float)
    Z_fit = double_shell_model(omega, constants, *popt)
    if 'k' in result.params and 'alpha' in result.params:
        Z_fit = Z_fit + Z_CPE(omega, result.params['k'], result.params['alpha'])

    # plot real  Impedance part
    plt.figure()
    plt.suptitle("double shell " + str(filename), y=1.05)
    plt.subplot(221)
    plt.xscale('log')
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
