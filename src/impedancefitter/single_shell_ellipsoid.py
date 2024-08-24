#    The ImpedanceFitter is a package to fit impedance spectra to
#    equivalent-circuit models using open-source software.
#
#    Copyright (C) 2023 Julius Zimmermann,
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

from impedancefitter.suspensionmodels import Lk, eps_sus_ellipsoid_MW


def eps_cell_single_shell_ellipsoid(omega, km, em, kcp, ecp, dm, Rcx, Rcy, Rcz):
    r"""Complex permittivity of single shell model.

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
    dm: double
        membrane thickness, value for :math:`d_\mathrm{m}`
    Rcx: double
        cell radius for x-semiaxis, value for :math:`R_\mathrm{cx}`
    Rcy: double
        cell radius for y-semiaxis, value for :math:`R_\mathrm{cy}`
    Rcz: double
        cell radius for z-semiaxis, value for :math:`R_\mathrm{cz}`

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Complex permittivity array
    """
    Rix = Rcx - dm
    Riy = Rcy - dm
    Riz = Rcz - dm

    v = Rix * Riy * Riz / (Rcx * Rcy * Rcz)

    epsi_cp = ecp - 1j * kcp / (e0 * omega)
    epsi_m = em - 1j * km / (e0 * omega)
    # model
    epsi_cell = []
    for i in range(3):
        Li_i = Lk(Rix, Riy, Riz, i)
        L_i = Lk(Rcx, Rcy, Rcz, i)
        epsi_tmp = epsi_m * (
            1.0
            + v * (epsi_cp - epsi_m) / (epsi_m + (epsi_cp - epsi_m) * (Li_i - v * L_i))
        )
        epsi_cell.append(epsi_tmp)
    return epsi_cell


def single_shell_ellipsoid_model(
    omega, km, em, kcp, ecp, kmed, emed, p, c0, dm, Rcx, Rcy, Rcz
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
    emed: double
        medium permittivity, value for :math:`\varepsilon_\mathrm{med}`
    kmed: double
        medium conductivity, value for :math:`\sigma_\mathrm{med}`
    p: double
        volume fraction
    dm: double
        membrane thickness, value for :math:`d_\mathrm{m}`
    Rcx: double
        cell radius for x-semiaxis, value for :math:`R_\mathrm{cx}`
    Rcy: double
        cell radius for y-semiaxis, value for :math:`R_\mathrm{cy}`
    Rcz: double
        cell radius for z-semiaxis, value for :math:`R_\mathrm{cz}`

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----
    .. warning::

        The unit capacitance is in pF!
        The cell radii and the membrane thickness have to be passed in um!

    """
    c0 *= 1e-12  # use pF as unit
    km *= 1e-6

    if dm < 1e-3:
        raise RuntimeError(
            "The membrane thickness is very small! It should be passed in um"
        )

    # cell model, contains three components
    epsi_cell = eps_cell_single_shell_ellipsoid(
        omega, km, em, kcp, ecp, dm, Rcx, Rcy, Rcz
    )

    epsi_med = emed - 1j * kmed / (e0 * omega)
    esus = eps_sus_ellipsoid_MW(epsi_med, *epsi_cell, p, Rcx, Rcy, Rcz)

    Ys = 1j * esus * omega * c0  # cell suspension admittance spectrum
    Z_fit = 1 / Ys

    return Z_fit
