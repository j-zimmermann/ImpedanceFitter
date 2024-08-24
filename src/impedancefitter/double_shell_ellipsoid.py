#    The ImpedanceFitter is a package to fit impedance spectra to equivalent-circuit models using open-source software.
#
#    Copyright (C) 2023 Julius Zimmermann, julius.zimmermann[AT]uni-rostock.de
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
from impedancefitter.single_shell_wall_ellipsoid import f


def eps_cell_double_shell_ellipsoid(omega, km, em, kcp, ecp, kne, ene, knp, enp, dm, dn, Rcx, Rcy, Rcz, Rnx, Rny, Rnz):
    r"""Complex permittivity of double shell model for ellipsoids

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
    dn: double
        cell nucleus membrane thickness, value for :math:`R_\mathrm{n}`
    Rcx: double
        cell radius for x-semiaxis, value for :math:`R_\mathrm{cx}`
    Rcy: double
        cell radius for y-semiaxis, value for :math:`R_\mathrm{cy}`
    Rcz: double
        cell radius for z-semiaxis, value for :math:`R_\mathrm{cz}`
    Rnx: double
        nucleus radius for x-semiaxis, value for :math:`R_\mathrm{nx}`
    Rny: double
        nucleus radius for y-semiaxis, value for :math:`R_\mathrm{ny}`
    Rnz: double
        nucleus radius for z-semiaxis, value for :math:`R_\mathrm{nz}`

    Notes
    -----

    The approach is based on [Asami2002]_, section 3.3.

    Returns
    -------

    :class:`numpy.ndarray`, complex
        Complex permittivity array

    """
    Rix = Rcx - dm
    Riy = Rcy - dm
    Riz = Rcz - dm

    Rnex = Rnx - dn
    Rney = Rny - dn
    Rnez = Rnz - dn

    v1 = (Rnex * Rney * Rnez / (Rnx * Rny * Rnz))
    v2 = (Rnx * Rny * Rnz / (Rix * Riy * Riz))
    v3 = (Rix * Riy * Riz / (Rcx * Rcy * Rcz))

    epsi_cp = ecp - 1j * kcp / (e0 * omega)
    epsi_m = em - 1j * km / (e0 * omega)
    epsi_np = enp - 1j * knp / (e0 * omega)
    epsi_ne = ene - 1j * kne / (e0 * omega)

    # model
    epsi_cell = []
    # inner-most shell: nucleus membrane and nucleoplasm
    for i in range(3):
        Li_i = Lk(Rnex, Rney, Rnez, i)
        L_i = Lk(Rnx, Rny, Rnz, i)
        epsi_tmp = f(epsi_ne, epsi_np, L_i, Li_i, v1)
        epsi_cell.append(epsi_tmp)
    # shell: nucleus membrane and cytoplasm
    for i in range(3):
        Li_i = Lk(Rnx, Rny, Rnz, i)
        L_i = Lk(Rix, Riy, Riz, i)
        epsi_cell[i] = f(epsi_cp, epsi_cell[i], L_i, Li_i, v2)
    # shell: cell membrane
    for i in range(3):
        Li_i = Lk(Rix, Riy, Riz, i)
        L_i = Lk(Rcx, Rcy, Rcz, i)
        epsi_cell[i] = f(epsi_m, epsi_cell[i], L_i, Li_i, v3)

    return epsi_cell


def double_shell_ellipsoid_model(omega, km, em, kcp, ecp, kmed, emed, kne, ene, knp, enp, p, c0, dm, dn, Rcx, Rcy, Rcz, Rnx, Rny, Rnz):
    r"""Impedance of single shell model

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
    dw: double
        cell wall thickness, value for :math:`R_\mathrm{c}`
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

    The approach is based on [Asami2002]_, section 3.3.

    See Also
    --------
    :meth:`impedancefitter.single_shell_ellipsoid.single_shell_ellipsoid_model`
    """
    c0 *= 1e-12  # use pF as unit
    km *= 1e-6
    kne *= 1e-3

    # cell model, contains three components
    epsi_cell = eps_cell_double_shell_ellipsoid(omega, km, em, kcp, ecp, kne, ene, knp, enp, dm, dn, Rcx, Rcy, Rcz, Rnx, Rny, Rnz)

    epsi_med = emed - 1j * kmed / (e0 * omega)
    esus = eps_sus_ellipsoid_MW(epsi_med, *epsi_cell, p, Rcx, Rcy, Rcz)

    Ys = 1j * esus * omega * c0  # cell suspension admittance spectrum
    Z_fit = 1 / Ys

    return Z_fit
