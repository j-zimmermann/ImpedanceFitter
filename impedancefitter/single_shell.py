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


from .elements import Z_CPE, Z_in, Z_loss
from scipy.constants import epsilon_0 as e0


def single_shell_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, cf, dm, Rc, k=None, alpha=None, L=None, C=None, R=None):
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

    .. todo::
        needs to be checked
    """
    c0 *= 1e-12  # use pF as unit
    cf *= 1e-12  # use pF as unit

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
    Zs_fit = 1 / Ys
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
