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


from .elements import Z_in, Z_CPE, e_sus, Z_sus, Z_loss


def cole_cole_model(omega, c0, el, tau, a, kdc, eh, k=None, alpha=None, L=None, C=None, R=None, cf=None):
    r"""
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

    """
    tau *= 1e-12  # use ps as unit
    c0 *= 1e-12  # use pF as unit
    if cf is not None:
        cf *= 1e-12  # use pF as unit
    es = e_sus(omega, eh, el, tau, a)
    Zs_fit = Z_sus(omega, es, kdc, c0)
    if k is not None and alpha is not None:
        Zep_fit = Z_CPE(omega, k, alpha)
        Zs_fit = Zs_fit + Zep_fit
    if L is not None:
        if C is None:
            Zin_fit = Z_in(omega, L, R)
        elif C is not None and R is not None:
            Zin_fit = Z_loss(omega, L, C, R)
        Zs_fit = Zs_fit + Zin_fit
    return Zs_fit
