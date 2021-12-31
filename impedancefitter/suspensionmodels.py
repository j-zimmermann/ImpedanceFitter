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


from .cubic_roots import get_cubic_roots
import numpy as np


def eps_sus_MW(epsi_med, epsi_p, p):
    r"""Maxwell-Wagner mixture model for dilute suspensions

    Parameters
    -----------
    epsi_med: :class:`numpy.ndarray`, complex
        complex permittivities of medium
    epsi_p: :class:`numpy.ndarray`, complex
        complex permittivities of suspension phase (cells, particles...)
    p: double
        volume fraction

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Complex permittivity array

    Notes
    -----

    The complex permittivity of the suspension $\varepsilon_\mathrm{sus}^\ast$ is given by

    .. math::
        \varepsilon_\mathrm{sus}^\ast = \varepsilon_\mathrm{med}^\ast\frac{(2\varepsilon_\mathrm{med}^\ast+\varepsilon_\mathrm{p}^\ast)-2p(\varepsilon_\mathrm{med}^\ast-\varepsilon_\mathrm{p}^\ast)}{(2\varepsilon_\mathrm{med}^\ast+\varepsilon_\mathrm{p}^\ast)+p(\varepsilon_\mathrm{med}^\ast-\varepsilon_\mathrm{p}^\ast)} \enspace ,

    with $\varepsilon_\mathrm{med}^\ast$ being the permittivity of the liquid medium and $\varepsilon_\mathrm{p}^\ast$ the permittivity of the suspended particle (e.g., cells).

    """

    return epsi_med * (((2. * epsi_med + epsi_p) - 2. * p * (epsi_med - epsi_p))
                       / ((2. * epsi_med + epsi_p) + p * (epsi_med - epsi_p)))


def F(root, epsi_p, epsi_med, p):
    return 1. / (1. - p) * (root - epsi_p) / (epsi_med - epsi_p) * (epsi_med / root)**(1. / 3.)


def bh_eps_model(epsi_med, epsi_p, p):
    r"""Complex permittvitiy of double shell model by Bruggeman-Hanai approach

    Parameters
    -----------
    epsi_med: :class:`numpy.ndarray`, complex
        Complex permittivity array of medium
    epsi_p: :class:`numpy.ndarray`, complex
        Complex permittivity array of suspended particles
    p: double
        volume fraction

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Complex permittivity array

    Notes
    -----

    The implementation follows [Cottet2019]_.
    Note that the approach here might not be numerically
    as accurate as the cubic equation solver (:meth:`impedancefitter.suspensionmodels.bhcubic_eps_model`).
    In the respective unit test, the deviation is below 1%, though.

    References
    ----------

    .. [Cottet2019] Cottet, J., Fabregue, O., Berger, C., Buret, F., Renaud, P., & Frénéa-Robin, M. (2019).
    MyDEP: A New Computational Tool for Dielectric Modeling of Particles and Cells. Biophysical Journal, 116(1), 12–18.
    https://doi.org/10.1016/j.bpj.2018.11.021

    See Also
    --------
    :meth:`impedancefitter.suspensionmodels.bhcubic_eps_model`

    """

    # initial values
    epsi_sus_n = np.copy(epsi_med)
    # solution
    epsi_sus_n1 = np.zeros(epsi_sus_n.shape, dtype=np.complex128)
    N = 200.  # use N steps, attention: hard-coded!
    h_p = p / N
    for i in range(int(N) + 1):
        epsi_sus_n1 = epsi_sus_n + h_p / (1. - h_p * float(i)) * (3. * epsi_sus_n * (epsi_p - epsi_sus_n)) / (2. * epsi_sus_n + epsi_p)
        epsi_sus_n = epsi_sus_n1
    return epsi_sus_n


def bhcubic_eps_model(epsi_med, epsi_p, p):
    r"""Complex permittvitiy of concentrated suspension model by Bruggeman-Hanai approach

    Parameters
    -----------
    epsi_med: :class:`numpy.ndarray`, complex
        Complex permittivity array of medium
    epsi_p: :class:`numpy.ndarray`, complex
        Complex permittivity array of suspended particles
    p: double
        volume fraction

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Complex permittivity array

    Notes
    -----

    The complex permittivity of the suspension $\varepsilon_\mathrm{sus}^\ast$ is given by [Hanai1979]_

    .. math::
        \frac{\varepsilon_\mathrm{sus}^\ast - \varepsilon_\mathrm{p}^\ast}{\varepsilon_\mathrm{med}^\ast - \varepsilon_\mathrm{p}^\ast} \left(\frac{\varepsilon_\mathrm{med}^\ast}{\varepsilon_\mathrm{sus}^\ast}\right)^{1/3} = 1 - p \enspace ,

    with $\varepsilon_\mathrm{med}^\ast$ being the permittivity of the liquid medium and $\varepsilon_\mathrm{p}^\ast$ the permittivity of the suspended particle (e.g., cells).

    Cubing the equation yields a cubic equation.
    The cubic roots (three in total) are the possible solutions for the complex permittivity
    of the suspension. Only one of them is physical, which can be found by substituting
    the cubic roots into another function [Hanai1979]_.

    A numerical solution is implemented in
    :meth:`impedancefitter.suspensionmodels.bh_eps_model`

    References
    ----------

    .. [Hanai1979] Hanai, T., Asami, K., & Korzumi, N. (1979).
                   Dielectric Theory of Concentrated Suspensions of Shell-Spheres in Particular Reference to the Analysis of Biological Cell Suspensions.
                   Bull. Inst. Chem. Res., Kyoto Univ, 57(4), 297–305.
                   http://hdl.handle.net/2433/76842

    See Also
    --------

    :meth:`impedancefitter.suspensionmodels.bh_eps_model`

    """

    epsi_sus = np.zeros(epsi_med.shape, dtype=np.complex128)
    for i in range(len(epsi_sus)):
        # Hanai: https://repository.kulib.kyoto-u.ac.jp/dspace/handle/2433/76842
        r1, r2, r3 = get_cubic_roots(-3. * epsi_p[i], (3. * np.power(epsi_p[i], 2.) + np.power((p - 1.) * (epsi_med[i] - epsi_p[i]), 3.) / epsi_med[i]), -np.power(epsi_p[i], 3.))
        for r in [r1, r2, r3]:
            Fvalue = F(r, epsi_p[i], epsi_med[i], p)
            if np.isclose(Fvalue.real, 1):
                epsi_sus[i] = r
                continue
    return epsi_sus
