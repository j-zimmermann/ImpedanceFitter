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
from .suspensionmodels import eps_sus_MW, bhcubic_eps_model


def particle_model(omega, ep, kp, kmed, emed, p, c0):
    r"""Single particle in suspension model

    Parameters
    -----------
    omega: :class:`numpy.ndarray`, double
        list of frequencies
    c0: double
        value for :math:`c_0`, unit capacitance in pF
    ep: double
        particle permittivity, value for :math:`\varepsilon_\mathrm{p}`
    kp: double
        particle conductivity, value for :math:`\sigma_\mathrm{p}` in :math:`\mu`\ S/m
    emed: double
        medium permittivity, value for :math:`\varepsilon_\mathrm{med}`
    kmed: double
        medium conductivity, value for :math:`\sigma_\mathrm{med}`
    p: double
        volume fraction

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----

    .. warning::

        The unit capacitance is in pF!
        The particle conductivity is in uF/m!

    """

    c0 *= 1e-12  # use pF as unit
    kp *= 1e-6

    epsi_med = emed - 1j * kmed / (e0 * omega)
    epsi_p = ep - 1j * kp / (e0 * omega)
    esus = eps_sus_MW(epsi_med, epsi_p, p)
    Ys = 1j * esus * omega * c0  # suspension admittance spectrum
    Z_fit = 1 / Ys

    return Z_fit


def particle_bh_model(omega, ep, kp, kmed, emed, p, c0):
    r"""Single particle in Bruggeman-Hanai suspension model

    Parameters
    -----------
    omega: :class:`numpy.ndarray`, double
        list of frequencies
    c0: double
        value for :math:`c_0`, unit capacitance in pF
    ep: double
        particle permittivity, value for :math:`\varepsilon_\mathrm{p}`
    kp: double
        particle conductivity, value for :math:`\sigma_\mathrm{p}` in :math:`\mu`\ S/m
    emed: double
        medium permittivity, value for :math:`\varepsilon_\mathrm{med}`
    kmed: double
        medium conductivity, value for :math:`\sigma_\mathrm{med}`
    p: double
        volume fraction

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----

    .. warning::

        The unit capacitance is in pF!
        The particle conductivity is in uF/m!
    """

    c0 *= 1e-12  # use pF as unit
    kp *= 1e-6

    epsi_med = emed - 1j * kmed / (e0 * omega)
    epsi_p = ep - 1j * kp / (e0 * omega)
    esus = bhcubic_eps_model(epsi_med, epsi_p, p)
    Ys = 1j * esus * omega * c0  # suspension admittance spectrum
    Z_fit = 1 / Ys

    return Z_fit
