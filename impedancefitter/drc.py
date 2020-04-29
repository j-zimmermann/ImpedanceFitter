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

from numpy import power


def drc_model(omega, RE, tauE, alpha, beta):
    """
    as described in 
    Emmert, S., Wolf, M., Gulich, R., Krohns, S., Kastner, S., Lunkenheimer, P., & Loidl, A. (2011). Electrode polarization effects in broadband dielectric spectroscopy. European Physical Journal B, 83(2), 157–165. https://doi.org/10.1140/epjb/e2011-20439-8
    """
    nom = power(1. + power(1j * omega * tauE, 1. - alpha), beta)
    return RE / nom 
