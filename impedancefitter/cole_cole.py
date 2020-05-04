#    The ImpedanceFitter is a package that provides means to fit impedance spectra to theoretical models using open-source software.
#
#    Copyright (C) 2018, 2019 Leonard Thiele, leonard.thiele[AT]uni-rostock.de
#    Copyright (C) 2018, 2019, 2020 Julius Zimmermann, julius.zimmermann[AT]uni-rostock.de
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


from .elements import e_sus, Z_sus


def cole_cole_model(omega, c0, el, tau, a, kdc, eh):
    r"""
Cole-Cole model as implemented in paper with DOI:10.1063/1.4737121.
                                     You need to provide the unit capacitance of your device to get the dielectric proper    ties of
                                     the Cole-Cole model.
   function holding the cole_cole_model equations, returning the calculated impedance
    Equations for calculations:

    .. math::

         Z_\mathrm{ep} = k^{-1} \* j\*\omega^{-\alpha}

    .. math::

        \varepsilon_\mathrm{s} = \varepsilon_\mathrm{h} + \frac{\varepsilon_\mathrm{l}-\varepsilon_\mathrm{h}}{1+(j\*\omega\*\tau)^a}

    .. math::

        Z_\mathrm{s} = \frac{1}{j\*\varepsilon_\mathrm{s}\*\omega\*c_\mathrm{0} + \frac{\sigma_\mathrm{dc}\*c_\mathrm{0}}{\varepsilon_\mathrm{0}} + j\*\omega\*c_\mathrm{f}}

    .. math::

        Z_\mathrm{fit} = Z_\mathrm{s} + Z_\mathrm{ep}

    find more details in formula 6 and 7 of https://ieeexplore.ieee.org/document/6191683

    +----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+
    | Parameter                        | Name in Script   | Description                                     | Physical Boundaries                                                                 |
    +==================================+==================+=================================================+=====================================================================================+
    | k                                | k                | constant phase element parameter                | has to be non 0, otherwise the function wil throw NAN or quit(1/k in the formula)   |
    +----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+
    | :math:`\alpha`                   | alpha            | constant phase element exponent                 | :math:`0<\alpha<1`                                                                  |
    +----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+
    | :math:`\varepsilon_\mathrm{l}`   | epsi\_l          | low frequency permitivity                       | :math:`el\geq1`                                                                     |
    +----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+
    | :math:`\varepsilon_\mathrm{h}`   | eh               | high frequency permitivity                      | :math:`eh\geq1`                                                                     |
    +----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+
    | :math:`\tau`                     | tau              | relaxation time                                 | :math:`\tau>0`                                                                      |
    +----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+
    | a                                | a                | exponent in formula for :math:`\varepsilon^*`   | :math:`0<a<1`                                                                       |
    +----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+
    | :math:`\sigma`                   | conductivity     | low frequency conductivity                      | :math:`\sigma > 0`                                                                  |
    +----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+

    """
    tau *= 1e-12  # use ps as unit
    c0 *= 1e-12  # use pF as unit
    es = e_sus(omega, eh, el, tau, a)
    Z_fit = Z_sus(omega, es, kdc, c0)

    return Z_fit


def cole_cole_R_model(omega, Rinf, R0, tau, a):
    r"""
                    Standard Cole-Cole circuit as given in paper with DOI:10.1016/b978-1-4832-3111-2.500    08-0.

    function holding the cole_cole_model equations, returning the calculated impedance
    Equations for calculations:

    .. math::

         Z_\mathrm{ep} = k^{-1} \* j\*\omega^{-\alpha}

    .. math::

        \Z_\mathrm{Cole} = R_\infty + \frac{R_0-R_\infty}{1+(j\*\omega\*\tau)^a}

    .. math::

        Z_\mathrm{fit} = Z_\mathrm{Cole} + Z_\mathrm{ep}

    """
    tau *= 1e-12  # use ps as unit
    Z_fit = Rinf + (R0 - Rinf) / (1. + 1j * omega * tau)**a
    return Z_fit
