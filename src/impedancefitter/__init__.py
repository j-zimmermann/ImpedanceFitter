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

logger = logging.getLogger("impedancefitter")
logger.addHandler(logging.NullHandler())


def set_logger(level=logging.INFO):
    """Set logging level."""
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
