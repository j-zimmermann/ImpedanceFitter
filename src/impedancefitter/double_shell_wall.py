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
from .double_shell import eps_cell_double_shell
from .suspensionmodels import eps_sus_MW, bhcubic_eps_model


def eps_cell_double_shell_wall(omega, km, em, kcp, ecp, kne, ene, knp, enp, kw, ew, dm, Rc, dn, Rn, dw):
    r"""Complex permittivity of double shell model with cell wall

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
        cell wall thickness, value for :math:`d_\mathrm{w}`

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Complex permittivity array

    Note
    ----

    Asami, K. (2002). Characterization of biological cells by dielectric spectroscopy. Journal of Non-Crystalline Solids, 305(1–3), 268–277. https://doi.org/10.1016/S0022-3093(02)01110-9

    """
    w = (1. - dw / (Rc + dw))**3

    epsi_w = ew - 1j * kw / (e0 * omega)
    epsi_p = eps_cell_double_shell(omega, km, em, kcp, ecp, kne, ene, knp, enp, dm, Rc, dn, Rn)
    # model
    epsi_cell = epsi_w * ((2. * epsi_w + epsi_p - 2. * w * (epsi_w - epsi_p))
                          / (2. * epsi_w + epsi_p + w * (epsi_w - epsi_p)))
    return epsi_cell


def double_shell_wall_model(omega, km, em, kcp, ecp, kne, ene, knp, enp, kw, ew, kmed, emed, p, c0, dm, Rc, dn, Rn, dw):
    r"""Impedance of double shell model with cell wall

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
    kw: double
        cell wall conductivity,  value for :math:`\sigma_\mathrm{w}` in S/m
    ew: double
        cell wall permittivity,  value for :math:`\varepsilon_\mathrm{w}`
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
    dw: double
        cell wall thickness

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----

    .. warning::

        The unit capacitance is in pF!

    """

    c0 *= 1e-12
    km *= 1e-6
    kne *= 1e-3

    epsi_med = emed + kmed / (1j * omega * e0)

    epsi_cell = eps_cell_double_shell_wall(omega, km, em, kcp, ecp, kne, ene, knp, enp, kw, ew, dm, Rc, dn, Rn, dw)
    esus = eps_sus_MW(epsi_med, epsi_cell, p)
    Ys = 1j * esus * omega * c0  # cell suspension admittance spectrum
    Z_fit = 1 / Ys
    return Z_fit


def double_shell_wall_bh_model(omega, km, em, kcp, ecp, kne, ene, knp, enp, kw, ew, kmed, emed, p, c0, dm, Rc, dn, Rn, dw):
    r"""Impedance of double shell model with cell wall using Bruggeman-Hanai approach

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
    kw: double
        cell wall conductivity,  value for :math:`\sigma_\mathrm{w}` in S/m
    ew: double
        cell wall permittivity,  value for :math:`\varepsilon_\mathrm{w}`
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
    dw: double
        wall thickness

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----

    .. warning::

        The unit capacitance is in pF!

    Note that here the Bruggeman-Hanai formula is used.

    See Also
    --------
    :meth:`impedancefitter.double_shell.double_shell_model`
    """

    c0 *= 1e-12
    km *= 1e-6
    kne *= 1e-3

    epsi_med = emed + kmed / (1j * omega * e0)

    epsi_cell = eps_cell_double_shell_wall(omega, km, em, kcp, ecp, kne, ene, knp, enp, kw, ew, dm, Rc, dn, Rn, dw)
    esus = bhcubic_eps_model(epsi_med, epsi_cell, p)
    Ys = 1j * esus * omega * c0  # cell suspension admittance spectrum
    Z_fit = 1 / Ys
    return Z_fit
