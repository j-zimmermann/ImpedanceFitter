#    The ImpedanceFitter is a package to fit impedance spectra to
#    equivalent-circuit models using open-source software.
#
#    Copyright (C) 2018, 2019 Leonard Thiele, leonard.thiele[AT]uni-rostock.de
#    Copyright (C) 2018, 2019, 2020 Julius Zimmermann,
#                                   julius.zimmermann[AT]uni-rostock.de
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
"""ImpedanceFitter is a package to fit impedance spectra."""

import logging

from .fitter import Fitter, set_logger
from .fra import (
    bode_csv_to_impedance,
    bode_to_impedance,
    open_short_compensation,
    read_bode_csv,
)
from .plotting import (
    emcee_plot,
    plot_admittance,
    plot_bode,
    plot_cole_cole,
    plot_compare_to_data,
    plot_comparison_dielectric_properties,
    plot_complex_permittivity,
    plot_dielectric_dispersion,
    plot_dielectric_modulus,
    plot_dielectric_properties,
    plot_impedance,
    plot_resistance_capacitance,
    plot_uncertainty,
)
from .postprocess import PostProcess
from .utils import (
    KK_integral_transform,
    available_file_format,
    available_models,
    draw_scheme,
    get_equivalent_circuit_model,
    get_labels,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = (
    "Fitter",
    "PostProcess",
    "get_labels",
    "available_models",
    "get_equivalent_circuit_model",
    "draw_scheme",
    "available_file_format",
    "KK_integral_transform",
    "plot_compare_to_data",
    "plot_impedance",
    "plot_dielectric_properties",
    "plot_dielectric_dispersion",
    "plot_dielectric_modulus",
    "plot_bode",
    "plot_cole_cole",
    "plot_complex_permittivity",
    "plot_admittance",
    "plot_comparison_dielectric_properties",
    "plot_uncertainty",
    "plot_resistance_capacitance",
    "emcee_plot",
    "bode_to_impedance",
    "bode_csv_to_impedance",
    "open_short_compensation",
    "read_bode_csv",
    "set_logger",
)
