#    The ImpedanceFitter is a package to fit impedance spectra to equivalent-circuit models using open-source software.
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
from .suspensionmodels import eps_sus_MW


def eps_cell_double_shell(omega, km, em, kcp, ecp, ene, kne, knp, enp, dm, Rc, dn, Rn):
    r"""Double Shell model.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`, double
        list of frequencies
        value for :math:`c_0`, unit capacitance in pF
    em: double
        membrane permittivity,membrane permittivity,  value for :math:`\varepsilon_\mathrm{m}`
    km: double
        membrane conductivity,  value for :math:`\sigma_\mathrm{m}` in :math:`\mu`\ S/m
    ecp: double
        cytoplasm permittivity,  value for :math:`\varepsilon_\mathrm{cp}`
    kcp: double
        cytoplasm conductivity,  value for :math:`\sigma_\mathrm{cp}`
    ene: double
        nuclear envelope permittivity,  value for :math:`\varepsilon_\mathrm{ne}`
    kne: double
        nuclear envelope conductivity,  value for :math:`\sigma_\mathrm{ne}` in mS/m
    enp: double
        nucleoplasm permittivity,  value for :math:`\varepsilon_\mathrm{np}`
    knp: double
        nucleoplasm conductivity,  value for :math:`\sigma_\mathrm{np}`
    dm: double
        membrane thickness, value for :math:`d_\mathrm{m}`
    Rc: double
        cell radius, value for :math:`R_\mathrm{c}`
    dn: double
        nuclear envelope thickness, value for :math:`d_\mathrm{n}`
    Rn: double
        nucleus radius, value for :math:`R_\mathrm{n}`


    Returns
    -------
    :class:`numpy.ndarray`, complex
        Permittivity array

    """


    v1 = (1. - dm / Rc)**3
    v2 = (Rn / (Rc - dm))**3
    v3 = (1. - dn / Rn)**3

    epsi_m = em + km / (1j * omega * e0)
    epsi_cp = ecp + kcp / (1j * omega * e0)
    epsi_ne = ene + kne / (1j * omega * e0)
    epsi_np = enp + knp / (1j * omega * e0)
    E3 = epsi_np / epsi_ne
    E2 = ((epsi_ne / epsi_cp) * (2. * (1. - v3) + (1. + 2. * v3) * E3)
          / ((2. + v3) + (1. - v3) * E3))
    E1 = ((epsi_cp / epsi_m) * (2. * (1. - v2) + (1. + 2. * v2) * E2)
          / ((2. + v2) + (1. - v2) * E2))

    epsi_cell = (epsi_m * (2. * (1. - v1) + (1. + 2. * v1) * E1)
                 / ((2. + v1) + (1. - v1) * E1))
    return epsi_cell

def double_shell_model(omega, km, em, kcp, ecp, ene, kne, knp, enp, kmed, emed, p, c0, dm, Rc, dn, Rn):
    r"""Double Shell model.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`, double
        list of frequencies
    c0: double
        value for :math:`c_0`, unit capacitance in pF
    em: double
        membrane permittivity,membrane permittivity,  value for :math:`\varepsilon_\mathrm{m}`
    km: double
        membrane conductivity,  value for :math:`\sigma_\mathrm{m}` in :math:`\mu`\ S/m
    ecp: double
        cytoplasm permittivity,  value for :math:`\varepsilon_\mathrm{cp}`
    kcp: double
        cytoplasm conductivity,  value for :math:`\sigma_\mathrm{cp}`
    ene: double
        nuclear envelope permittivity,  value for :math:`\varepsilon_\mathrm{ne}`
    kne: double
        nuclear envelope conductivity,  value for :math:`\sigma_\mathrm{ne}` in mS/m
    enp: double
        nucleoplasm permittivity,  value for :math:`\varepsilon_\mathrm{np}`
    knp: double
        nucleoplasm conductivity,  value for :math:`\sigma_\mathrm{np}`
    emed: double
        medium permittivity,  value for :math:`\varepsilon_\mathrm{med}`
    kmed: double
        medium conductivity,  value for :math:`\sigma_\mathrm{med}`
    p: double
        volume fraction
    dm: double
        membrane thickness, value for :math:`d_\mathrm{m}`
    Rc: double
        cell radius, value for :math:`R_\mathrm{c}`
    dn: double
        nuclear envelope thickness, value for :math:`d_\mathrm{n}`
    Rn: double
        nucleus radius, value for :math:`R_\mathrm{n}`


    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----

    .. warning::

        The unit capacitance is in pF!

    Equations for the single-shell-model [Feldman2003]_:

    .. math::

            \nu_1 = \left(1-\frac{d_\mathrm{m}}{R_\mathrm{c}}\right)^3

    .. math::

            \nu_\mathrm{2} = \left(\frac{R_\mathrm{n}}{R-d}\right)^3

    .. math::

            \nu_\mathrm{3} = \left(1-\frac{d_\mathrm{n}}{R_\mathrm{n}}\right)^3

    .. math::

            \varepsilon_\mathrm{c}^\ast = \varepsilon_\mathrm{m}^\ast\frac{2(1-\nu_\mathrm{1})+(1+2\nu_\mathrm{1})E_\mathrm{1}}{(2+\nu_\mathrm{1})+(1-\nu_\mathrm{1})E_\mathrm{1}}

    .. math::

            E_\mathrm{1}  = \frac{\varepsilon_\mathrm{cp}^\ast}{\varepsilon_\mathrm{m}^\ast} \frac{2(1-\nu_\mathrm{2})+(1+2\nu_\mathrm{2})E_\mathrm{2}}{(2+\nu_\mathrm{2})+(1-\nu_\mathrm{2})E_\mathrm{2}}

    .. math::

            E_\mathrm{2} = \frac{\varepsilon_\mathrm{ne}^\ast}{\varepsilon_\mathrm{cp}^\ast}\frac{2(1-\nu_\mathrm{3})+(1+2\nu_\mathrm{3})E_\mathrm{3}}{(2+\nu_\mathrm{3})+(1-\nu_\mathrm{3})E_\mathrm{3}}

    .. math::

            E_\mathrm{3} = \frac{\varepsilon_\mathrm{np}^\ast}{\varepsilon_\mathrm{ne}^\ast}


    .. math::

        \varepsilon_\mathrm{m} = \varepsilon_\mathrm{m} - j \frac{\sigma_\mathrm{m}}{\varepsilon_0 \omega}

    .. math::

        \varepsilon_\mathrm{cp} = \varepsilon_\mathrm{cp} - j \frac{\sigma_\mathrm{cp}}{\varepsilon_0 \omega}

    .. math::

        \varepsilon_\mathrm{ne} = \varepsilon_\mathrm{ne} - j \frac{\sigma_\mathrm{ne}}{\varepsilon_0 \omega}


    .. math::

        \varepsilon_\mathrm{np} = \varepsilon_\mathrm{np} - j \frac{\sigma_\mathrm{np}}{\varepsilon_0 \omega}


    .. math::

        \varepsilon_\mathrm{cell}^\ast = \varepsilon_\mathrm{m}^\ast \frac{2 (1 - \nu_1) + (1 + 2 \nu_1) E_1}{(2 + \nu_1) + (1 - \nu_1) E_1}

    .. math::

        \varepsilon_\mathrm{sus}^\ast = \varepsilon_\mathrm{med}^\ast
        \frac{(2 \varepsilon_\mathrm{med}^\ast + \varepsilon_\mathrm{cell}^\ast) - 2 p
        (\varepsilon_\mathrm{med}^\ast - \varepsilon_\mathrm{cell}^\ast)}
        {(2 \varepsilon_\mathrm{med}^\ast + \varepsilon_\mathrm{cell}^\ast) + p
        (\varepsilon_\mathrm{med}^\ast - \varepsilon_\mathrm{cell}^\ast)}

    .. math::

        Z = \frac{1}{j \varepsilon_\mathrm{sus}^\ast \omega c_0}

    In [Ermolina2000]_ , there have been reported upper/lower limits for certain parameters.
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

    .. [Ermolina2000] Ermolina, I., Polevaya, Y., & Feldman, Y. (2000). Analysis of dielectric spectra of eukaryotic cells by computer modeling. European Biophysics Journal, 29(2), 141–145. https://doi.org/10.1007/s002490050259

    See Also
    --------
    :meth:`impedancefitter.single_shell.single_shell_model`
    """

    c0 *= 1e-12
    km *= 1e-6
    kne *= 1e-3

    epsi_med = emed + kmed / (1j * omega * e0)

    epsi_cell = eps_cell_double_shell(omega, km, em, kcp, ecp, ene, kne, knp, enp, dm, Rc, dn, Rn)
    esus = eps_sus_MW(epsi_med, epsi_cell, p)
    Ys = 1j * esus * omega * c0  # cell suspension admittance spectrum
    Z_fit = 1 / Ys
    return Z_fit
