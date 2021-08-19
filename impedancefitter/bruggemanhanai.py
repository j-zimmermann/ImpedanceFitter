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
from .single_shell import eps_cell, eps_sus
from .cubic_roots import get_cubic_roots
import numpy as np


def bh_eps_single_shell_model(omega, em, km, kcp, ecp, kmed, emed, p, dm, Rc):
    r"""Complex permittvitiy of single shell model by Bruggeman-Hanai approach

    Parameters
    -----------
    omega: :class:`numpy.ndarray`, double
        list of frequencies
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
    Rc: double
        cell radius, value for :math:`R_\mathrm{c}`

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Complex permittivity array

    Notes
    -----

    The implementation follows [Cottet2019]_.

    References
    ----------

    .. [Cottet2019] Cottet, J., Fabregue, O., Berger, C., Buret, F., Renaud, P., & Frénéa-Robin, M. (2019).
    MyDEP: A New Computational Tool for Dielectric Modeling of Particles and Cells. Biophysical Journal, 116(1), 12–18.
    https://doi.org/10.1016/j.bpj.2018.11.021

    See Also
    --------
    :meth:`impedancefitter.single_shell.single_shell_model`
    """

    if p < 0.1:
        raise RuntimeError("Volume fraction is less than 10%. Use the standard single shell model in this case!")

    epsi_med = emed - 1j * kmed / (e0 * omega)
    epsi_cell = eps_cell(omega, em, km, kcp, ecp, dm, Rc)
    epsi_c = e0 * eps_cell(omega, em, km, kcp, ecp, dm, Rc)
    epsi_sus_n = e0 * emed - 1j * kmed / omega
    epsi_sus_n1 = np.zeros(epsi_sus_n.shape, dtype=np.complex128)
    N = 150.  # use 150 steps, attention: hard-coded!
    h_p = p / N
    for i in range(int(N) + 1):
        if i == 0:
            assert np.all(np.isclose(epsi_sus_n / e0, epsi_med))
        # epsi_sus_n = epsi_sus_n + h_p / (1. - h_p * float(i)) * (3. * epsi_sus_n * (epsi_cell - epsi_sus_n)) / (2. * epsi_sus_n + epsi_cell)
        epsi_sus_n1 = epsi_sus_n + h_p / (1. - h_p * float(i)) * (3. * epsi_sus_n * (epsi_c - epsi_sus_n)) / (2. * epsi_sus_n + epsi_c)
        epsi_sus_n = epsi_sus_n1
    return epsi_sus_n / e0


def F(root, epsi_cell, epsi_med, p):
    return 1. / (1. - p) * (root - epsi_cell) / (epsi_med - epsi_cell) * (epsi_med / root)**(1. / 3.)

def bhcubic_eps_single_shell_model(omega, em, km, kcp, ecp, kmed, emed, p, dm, Rc):
    r"""Complex permittvitiy of single shell model by Bruggeman-Hanai approach

    Parameters
    -----------
    omega: :class:`numpy.ndarray`, double
        list of frequencies
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
    Rc: double
        cell radius, value for :math:`R_\mathrm{c}`

    Returns
    -------
    :class:`numpy.ndarray`, complex
        Complex permittivity array

    Notes
    -----

    The implementation follows [Cottet2019]_.

    References
    ----------

    .. [Cottet2019] Cottet, J., Fabregue, O., Berger, C., Buret, F., Renaud, P., & Frénéa-Robin, M. (2019).
    MyDEP: A New Computational Tool for Dielectric Modeling of Particles and Cells. Biophysical Journal, 116(1), 12–18.
    https://doi.org/10.1016/j.bpj.2018.11.021

    See Also
    --------
    :meth:`impedancefitter.single_shell.single_shell_model`
    """

    """
    if p < 0.1:
        raise RuntimeError("Volume fraction is less than 10%. Use the standard single shell model in this case!")
    """

    epsi_sus = np.zeros(omega.shape, dtype=np.complex128)
    epsi_med = emed - 1j * kmed / (e0 * omega)
    epsi_cell = eps_cell(omega, em, km, kcp, ecp, dm, Rc)
    for i in range(len(omega)):
        # Cotted description, typos?
        # r1, r2, r3 = get_cubic_roots(-3. * epsi_cell[i], 3. * (np.power(epsi_cell[i], 2.) + (p - 1.)**3 * np.power(epsi_med[i] - epsi_cell[i], 3.) / epsi_med[i]), -np.power(epsi_cell[i], 3.))
        # Hanai: https://repository.kulib.kyoto-u.ac.jp/dspace/handle/2433/76842
        r1, r2, r3 = get_cubic_roots(-3. * epsi_cell[i], (3. * np.power(epsi_cell[i], 2.) + np.power((p - 1.) * (epsi_med[i] - epsi_cell[i]), 3.) / epsi_med[i]), -np.power(epsi_cell[i], 3.))
        # print(r1, r2, r3)
        for r in [r1, r2, r3]:
            Fvalue = F(r, epsi_cell[i], epsi_med[i], p)
            # print(Fvalue)
            if np.isclose(Fvalue.real, 1):
                epsi_sus[i] = r
                continue
    return epsi_sus


def bhcubic_eps_suspension_model(omega, ep, kp, kmed, emed, p):
    r"""Complex permittvitiy of suspension model by Bruggeman-Hanai approach

    Parameters
    -----------
    omega: :class:`numpy.ndarray`, double
        list of frequencies
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
        Complex permittivity array

    Notes
    -----

    The implementation follows .

    References
    ----------

    """

    """
    if p < 0.1:
        raise RuntimeError("Volume fraction is less than 10%. Use the standard single shell model in this case!")
    """

    epsi_sus = np.zeros(omega.shape, dtype=np.complex128)
    epsi_med = emed - 1j * kmed / (e0 * omega)
    epsi_p = ep - 1j * kp / (e0 * omega)
    for i in range(len(omega)):
        # Hanai: https://repository.kulib.kyoto-u.ac.jp/dspace/handle/2433/76842
        r1, r2, r3 = get_cubic_roots(-3. * epsi_p[i], (3. * np.power(epsi_p[i], 2.) + np.power((p - 1.) * (epsi_med[i] - epsi_p[i]), 3.) / epsi_med[i]), -np.power(epsi_p[i], 3.))
        # print(r1, r2, r3)
        for r in [r1, r2, r3]:
            Fvalue = F(r, epsi_p[i], epsi_med[i], p)
            # print(Fvalue)
            if np.isclose(Fvalue.real, 1):
                epsi_sus[i] = r
                continue
    return epsi_sus
