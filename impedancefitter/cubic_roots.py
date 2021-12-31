#    The ImpedanceFitter is a package to fit impedance spectra to equivalent-circuit models using open-source software.
#
#    Copyright (C) 2021, 2022 Julius Zimmermann, julius.zimmermann[AT]uni-rostock.de
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

from cmath import sqrt
import numpy as np


def get_cubic_roots(a, b, c):
    """Get the roots of a cubic equation.

    Parameters
    -----------

    a: complex
        polynomial coefficient
    b: complex
        polynomial coefficient
    c: complex
        polynomial coefficient

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Complex permittivity array

    Notes
    -----

    Return the roots of an expression

    .. math::

        z^3 + a z^2 + b z + c = 0

    More details can be found in [Press2007]_.

    References
    ----------

    .. [Press2007]  Press, W.H., Teukolsky, S.A., Vetterling, W.T., and Flannery, B.P. (2007)
                    Numerical recipes : the art of scientific computing.
                    Cambridge Univ. Press, USA, 3rd edition

    """
    Q = (a * a - 3. * b) / 9.
    R = (2. * a * a * a - 9. * a * b + 27. * c) / 54.
    A = -(R + sqrt(R * R - Q * Q * Q))**(1. / 3.)
    if np.isclose(abs(A), 0):
        B = 0
    else:
        B = Q / A
    # first root
    x1 = (A + B) - a / 3.
    x2 = -0.5 * (A + B) - a / 3. + 1j * 0.5 * sqrt(3.) * (A - B)
    x3 = -0.5 * (A + B) - a / 3. - 1j * 0.5 * sqrt(3.) * (A - B)
    return x1, x2, x3
