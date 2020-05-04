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

from scipy.constants import epsilon_0 as e0


def double_shell_model(omega, km, em, kcp, ecp, ene, kne, knp, enp, kmed, emed, p, c0, dm, Rc, dn, Rn):
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


    In [Erm2000]_ , there have been reported upper/lower limits for certain parameters.
    They could act as a first guess for the bounds of the optimization method.

    +-----------------------------------+---------------+---------------+
    | Parameter                         | lower limit   | upper limit   |
    +===================================+===============+===============+
    | :math:`\varepsilon_\mathrm{m}`    | 1.4           | 16.8          |
    +-----------------------------------+---------------+---------------+
    | :math:`\sigma_\mathrm{m}`         | 8e-8          | 5.6e-5        |
    +-----------------------------------+---------------+---------------+
    | :math:`\varepsilon_\mathrm{cp}`   | 60            | 77            |
    +-----------------------------------+---------------+---------------+
    | :math:`\sigma_\mathrm{cp}`        | 0.033         | 1.1           |
    +-----------------------------------+---------------+---------------+
    | :math:`\varepsilon_\mathrm{ne}`   | 6.8           | 100           |
    +-----------------------------------+---------------+---------------+
    | :math:`\sigma_\mathrm{ne}`        | 8.3e-5        | 7e-3          |
    +-----------------------------------+---------------+---------------+
    | :math:`\varepsilon_\mathrm{np}`   | 32            | 300           |
    +-----------------------------------+---------------+---------------+
    | :math:`\sigma_\mathrm{np}`        | 0.25          | 2.2           |
    +-----------------------------------+---------------+---------------+
    | R                                 | 3.5e-6        | 10.5e-6       |
    +-----------------------------------+---------------+---------------+
    | :math:`R_\mathrm{n}`              | 2.95e-6       | 8.85e-6       |
    +-----------------------------------+---------------+---------------+
    | d                                 | 3.5e-9        | 10.5e-9       |
    +-----------------------------------+---------------+---------------+
    | :math:`d_\mathrm{n}`              | 2e-8          | 6e-8          |
    +-----------------------------------+---------------+---------------+

    .. [Erm2000] Ermolina, I., Polevaya, Y., & Feldman, Y. (2000). Analysis of dielectric spectra of eukaryotic cells by computer modeling. European Biophysics Journal, 29(2), 141–145. https://doi.org/10.1007/s002490050259

    .. todo::
        needs to be checked

    """
    c0 *= 1e-12
    v1 = (1. - dm / Rc)**3
    v2 = (Rn / (Rc - dm))**3
    v3 = (1. - dn / Rn)**3

    epsi_m = em + km / (1j * omega * e0)
    epsi_cp = ecp + kcp / (1j * omega * e0)
    epsi_ne = ene + kne / (1j * omega * e0)
    epsi_np = enp + knp / (1j * omega * e0)
    epsi_med = emed + kmed / (1j * omega * e0)

    E3 = epsi_np / epsi_ne
    E2 = ((epsi_ne / epsi_cp) * (2. * (1. - v3) + (1. + 2. * v3) * E3)
          / ((2. + v3) + (1. - v3) * E3))  # Eq. 13
    E1 = ((epsi_cp / epsi_m) * (2. * (1. - v2) + (1. + 2. * v2) * E2)
          / ((2. + v2) + (1. - v2) * E2))  # Eq. 14

    epsi_cell = (epsi_m * (2. * (1. - v1) + (1. + 2. * v1) * E1)
                 / ((2. + v1) + (1. - v1) * E1))  # Eq. 11
    E0 = epsi_cell / epsi_med
    esus = epsi_med * (2. * (1. - p) + (1. + 2. * p) * E0) / ((2. + p) + (1. - p) * E0)
    Ys = 1j * esus * omega * c0  # cell suspension admittance spectrum
    Z_fit = 1 / Ys
    return Z_fit
