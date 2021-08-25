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


def eps_cell_single_shell(omega, em, km, kcp, ecp, dm, Rc):
    r"""Single Shell model

    Parameters
    -----------
    omega: :class:`numpy.ndarray`, double
        list of frequencies
    em: double
        membrane permittivity, value for :math:`\varepsilon_\mathrm{m}`
    km: double
        membrane conductivity, value for :math:`\sigma_\mathrm{m}`
    ecp: double
        cytoplasm permittivity, value for :math:`\varepsilon_\mathrm{cp}`
    kcp: double
        cytoplasm conductivity, value for :math:`\sigma_\mathrm{cp}`
    dm: double
        membrane thickness, value for :math:`d_\mathrm{m}`
    Rc: double
        cell radius, value for :math:`R_\mathrm{c}`

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Complex permittivity array
    """
    v1 = (1. - dm / Rc)**3

    epsi_cp = ecp - 1j * kcp / (e0 * omega)
    epsi_m = em - 1j * km / (e0 * omega)
    # model
    E1 = epsi_cp / epsi_m
    epsi_cell = epsi_m * (2. * (1. - v1) + (1. + 2. * v1) * E1) / ((2. + v1) + (1. - v1) * E1)
    return epsi_cell


def single_shell_model(omega, em, km, kcp, ecp, kmed, emed, p, c0, dm, Rc):
    r"""Single Shell model

    Parameters
    -----------
    omega: :class:`numpy.ndarray`, double
        list of frequencies
    c0: double
        value for :math:`c_0`, unit capacitance in pF
    em: double
        membrane permittivity, value for :math:`\varepsilon_\mathrm{m}`
    km: double
        membrane conductivity, value for :math:`\sigma_\mathrm{m}` in :math:`\mu`\ S/m
    ecp: double
        cytoplasm permittivity, value for :math:`\varepsilon_\mathrm{cp}`
    kcp: double
        cytoplasm conductivity, value for :math:`\sigma_\mathrm{cp}`
    emed: double
        medium permittivity, value for :math:`\varepsilon_\mathrm{med}`
    kmed: double
        medium conductivity, value for :math:`\sigma_\mathrm{med}`
    p: double
        volume fraction
    dm: double
        membrane thickness, value for :math:`d_\mathrm{m}`
    Rc: double
        cell radius, value for :math:`R_\mathrm{c}`

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

        \varepsilon_\mathrm{m} = \varepsilon_\mathrm{m} - j \frac{\sigma_\mathrm{m}}{\varepsilon_0 \omega}

    .. math::

        \varepsilon_\mathrm{cp} = \varepsilon_\mathrm{cp} - j \frac{\sigma_\mathrm{cp}}{\varepsilon_0 \omega}

    .. math::

        \varepsilon_\mathrm{cell}^\ast = \varepsilon_\mathrm{m}^\ast \frac{2 (1 - \nu_1) + (1 + 2 \nu_1) E_1}{(2 + \nu_1) + (1 - \nu_1) E_1}

    .. math::

        E_1 = \frac{\varepsilon_\mathrm{cp}^\ast}{\varepsilon_\mathrm{m}^\ast}

    .. math::

        \varepsilon_\mathrm{sus}^\ast = \varepsilon_\mathrm{med}^\ast
        \frac{(2 \varepsilon_\mathrm{med}^\ast + \varepsilon_\mathrm{cell}^\ast) - 2 p
        (\varepsilon_\mathrm{med}^\ast - \varepsilon_\mathrm{cell}^\ast)}
        {(2 \varepsilon_\mathrm{med}^\ast + \varepsilon_\mathrm{cell}^\ast) + p
        (\varepsilon_\mathrm{med}^\ast - \varepsilon_\mathrm{cell}^\ast)}

    .. math::

        Z = \frac{1}{j \varepsilon_\mathrm{sus}^\ast \omega c_0}

    References
    ----------

    .. [Feldman2003] Feldman, Y., Ermolina, I., & Hayashi, Y. (2003).
           Time domain dielectric spectroscopy study of biological systems.
           IEEE Transactions on Dielectrics and Electrical Insulation, 10, 728–753.
           https://doi.org/10.1109/TDEI.2003.1237324

    See Also
    --------
    :meth:`impedancefitter.double_shell.double_shell_model`
    """
    c0 *= 1e-12  # use pF as unit
    km *= 1e-6

    # cell model
    epsi_cell = eps_cell(omega, em, km, kcp, ecp, dm, Rc)

    epsi_med = emed - 1j * kmed / (e0 * omega)
    # electrode polarization and calculation of Z
    esus = eps_sus_MW(epsi_med, epsi_cell, p)
    Ys = 1j * esus * omega * c0  # cell suspension admittance spectrum
    Z_fit = 1 / Ys

    return Z_fit
