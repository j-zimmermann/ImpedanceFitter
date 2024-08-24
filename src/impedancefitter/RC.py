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

from numpy import power
from scipy.constants import epsilon_0 as e0


def RC_model(omega, Rd, Cd):
    """Simple RC model, resistor in parallel with capacitor.

    Parameters
    ----------

    omega: :class:`numpy.ndarray`, double
        list of frequencies
    Rd: double
        Resistance.
    Cd: double
        Capacitance

    Notes
    -----
    .. warning::

        `Cd` is in pF!

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array
    """
    Cfit = Cd * 1e-12
    Z_fit = Rd / (1. + 1j * omega * Cfit * Rd)
    return Z_fit


def rc_model(omega, c0, kdc, eps):
    """Simple RC model of a lossy dielectric to obtain dielectric properties.

    Parameters
    -----------

    omega: :class:`numpy.ndarray`, double
        list of frequencies
    c0: double
        unit capacitance in pF
    eps: double
        relative permittivity
    kdc: double
        conductivity


    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----

    .. warning::

        C0 is in pF!

    """
    c0 *= 1e-12
    Rd = e0 / (kdc * c0)
    # Cd = eps * c0
    factor = eps * e0 / kdc  # Cd * Rd
    Z_fit = Rd / (1. + 1j * omega * factor)
    return Z_fit


def drc_model(omega, RE, tauE, alpha, beta):
    """Distributed RC circuit.

    Parameters
    -----------

    omega: :class:`numpy.ndarray`, double
        list of frequencies
    RE: double
        resistance
    tauE: double
        relaxation time, in ns
    alpha: double
        Cole-Cole exponent
    beta: double
        DRC exponent, beta = 1 equals Cole-Cole model

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----
    Described for example in [Emmert2011]_.

    .. warning::

        The time constant tauE is in ns!


    References
    ----------

    .. [Emmert2011] Emmert, S., Wolf, M., Gulich, R., Krohns, S., Kastner, S., Lunkenheimer, P., & Loidl, A. (2011).
                    Electrode polarization effects in broadband dielectric spectroscopy.
                    European Physical Journal B, 83(2), 157–165.
                    https://doi.org/10.1140/epjb/e2011-20439-8
    """
    tauE *= 1e-9
    nom = power(1. + power(1j * omega * tauE, 1. - alpha), beta)
    return RE / nom


def rc_tau_model(omega, Rk, tauk):
    r"""Compute RC model without explicit capacitance.

    Parameters
    -----------

    omega: :class:`numpy.ndarray`, double
        list of frequencies
    Rk: double
        resistance
    tauk: double
        relaxation time

    Notes
    -----
    Returns
    -------
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----
    Used for example in the Lin-KK test [Schoenleber2014]_.

    The impedance reads

    .. math::

        Z = \frac{R_\mathrm{k}}{1+ j \omega \tau_\mathrm{k}}

    """

    return Rk / (1. + 1j * omega * tauk)
