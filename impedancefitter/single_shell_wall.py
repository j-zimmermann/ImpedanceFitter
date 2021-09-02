#    The ImpedanceFitter is a package to fit impedance spectra to equivalent-circuit models using open-source software.
#
#    Copyright (C) 2021 Julius Zimmermann, julius.zimmermann[AT]uni-rostock.de
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
from .single_shell import eps_cell_single_shell



def eps_cell_single_shell_wall(omega, em, km, kcp, ecp, ew, kw, dm, Rc, dw):
    r"""Single shell model with cell wall

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
    ew: double
        cell wall permittivity, value for :math:`\varepsilon_\mathrm{w}`
    kw: double
        cell wall conductivity, value for :math:`\sigma_\mathrm{w}`
    dm: double
        membrane thickness, value for :math:`d_\mathrm{m}`
    Rc: double
        cell radius, value for :math:`R_\mathrm{c}`
    dw: double
        cell wall thickness, value for :math:`R_\mathrm{c}`

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Complex permittivity array

    Note
    ----

    Asami, K. (2002). Characterization of biological cells by dielectric spectroscopy. Journal of Non-Crystalline Solids, 305(1–3), 268–277. https://doi.org/10.1016/S0022-3093(02)01110-9

    """
    w = (1. - dw / Rc)

    epsi_w = ew - 1j * kw / (e0 * omega)
    epsi_p = eps_cell_single_shell(omega, em, km, kcp, ecp, dm, Rc)
    # model
    epsi_cell = epsi_w * ((2. * epsi_w + epsi_p - 2. * w * (epsi_w - epsi_p))
                          / (2. * epsi_w + epsi_p + w * (epsi_w - epsi_p)))
    return epsi_cell
