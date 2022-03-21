#    The ImpedanceFitter is a package to fit impedance spectra to equivalent-circuit models using open-source software.
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

import logging
logger = logging.getLogger('impedancefitter')
logger.addHandler(logging.NullHandler())


def set_logger(level=logging.INFO):
    logger.setLevel(level)
    # to avoid multiple output in Jupyter notebooks
    if len(logger.handlers) == 1:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        logger.addHandler(ch)
    else:
        for handler in logger.handlers:
            if type(handler) == logging.StreamHandler:
                handler.setLevel(level)


from .fitter import Fitter
from .postprocess import PostProcess
from .utils import get_labels, available_models, get_equivalent_circuit_model, draw_scheme, available_file_format, KK_integral_transform
from .plotting import plot_compare_to_data, plot_impedance, plot_dielectric_properties, emcee_plot, plot_bode, plot_cole_cole, plot_complex_permittivity, plot_admittance, plot_dielectric_dispersion, plot_dielectric_modulus
from .fra import bode_to_impedance, bode_csv_to_impedance, parallel, open_short_compensation, read_bode_csv
from .__version__ import __version__
