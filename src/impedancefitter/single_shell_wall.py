#    The ImpedanceFitter is a package to fit impedance spectra to
#    equivalent-circuit models using open-source software.
#
#    Copyright (C) 2021 Julius Zimmermann,
#                                   julius.zimmermann[AT]uni-rostock.de
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

from .single_shell import eps_cell_single_shell
from .suspensionmodels import bhcubic_eps_model, eps_sus_MW


def eps_cell_single_shell_wall(omega, km, em, kcp, ecp, kw, ew, dm, Rc, dw):
    r"""Single shell model with cell wall.

    Parameters
    ----------
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

    Asami, K. (2002).
    Characterization of biological cells by dielectric spectroscopy.
    Journal of Non-Crystalline Solids, 305(1–3), 268–277.
    https://doi.org/10.1016/S0022-3093(02)01110-9

    """
    w = (1.0 - dw / (Rc + dw)) ** 3

    epsi_w = ew - 1j * kw / (e0 * omega)
    epsi_p = eps_cell_single_shell(omega, km, em, kcp, ecp, dm, Rc)
    # model
    epsi_cell = epsi_w * (
        (2.0 * epsi_w + epsi_p - 2.0 * w * (epsi_w - epsi_p))
        / (2.0 * epsi_w + epsi_p + w * (epsi_w - epsi_p))
    )
    return epsi_cell


def single_shell_wall_model(
    omega, km, em, kcp, ecp, kw, ew, kmed, emed, p, c0, dm, Rc, dw
):
    r"""Impedance of single shell model.

    Parameters
    ----------
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
    ew: double
        cell wall permittivity, value for :math:`\varepsilon_\mathrm{w}`
    kw: double
        cell wall conductivity, value for :math:`\sigma_\mathrm{w}`
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
    dw: double
        cell wall thickness, value for :math:`d_\mathrm{w}`

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----
    .. warning::

        The unit capacitance is in pF!
        The membrane conductivity is in uS/m!

    See Also
    --------
    :meth:`impedancefitter.single_shell.single_shell_wall_model`
    """
    c0 *= 1e-12  # use pF as unit
    km *= 1e-6

    # cell model
    epsi_cell = eps_cell_single_shell_wall(omega, km, em, kcp, ecp, kw, ew, dm, Rc, dw)

    epsi_med = emed - 1j * kmed / (e0 * omega)
    esus = eps_sus_MW(epsi_med, epsi_cell, p)
    Ys = 1j * esus * omega * c0  # cell suspension admittance spectrum
    Z_fit = 1 / Ys

    return Z_fit


def single_shell_wall_bh_model(
    omega, km, em, kcp, ecp, kw, ew, kmed, emed, p, c0, dm, Rc, dw
):
    r"""Impedance of single shell model using Bruggeman-Hanai approach.

    Parameters
    ----------
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
    ew: double
        cell wall permittivity, value for :math:`\varepsilon_\mathrm{w}`
    kw: double
        cell wall conductivity, value for :math:`\sigma_\mathrm{w}`
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
    dw: double
        cell wall thickness, value for :math:`d_\mathrm{w}`

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----
    .. warning::

        The unit capacitance is in pF!
        The membrane conductivity is in uS/m!

    See Also
    --------
    :meth:`impedancefitter.single_shell.single_shell_wall_model`
    """
    c0 *= 1e-12  # use pF as unit
    km *= 1e-6

    # cell model
    epsi_cell = eps_cell_single_shell_wall(omega, km, em, kcp, ecp, kw, ew, dm, Rc, dw)

    epsi_med = emed - 1j * kmed / (e0 * omega)
    esus = bhcubic_eps_model(epsi_med, epsi_cell, p)
    Ys = 1j * esus * omega * c0  # cell suspension admittance spectrum
    Z_fit = 1 / Ys

    return Z_fit
