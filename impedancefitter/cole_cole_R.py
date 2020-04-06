#    The ImpedanceFitter is a package that provides means to fit impedance spectra to theoretical models using open-source software.
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


def cole_cole_R_model(omega, Rinf, R0, tau, a):
    r"""
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
