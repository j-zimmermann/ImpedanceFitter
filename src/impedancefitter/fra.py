#    The ImpedanceFitter is a package to fit impedance spectra to
#
#    equivalent-circuit models using open-source software.
#
#    Copyright (C) 2021, 2023 Henning Bathel, henning.bathel2[AT]uni-rostock.de
#    Copyright (C) 2021, 2023 Julius Zimmermann,
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

import json
import logging
import os

import numpy as np
import pandas

"""
Collection of useful functions to get Impedance from Bode Diagram CSV files
measured with frequency response analysers (FRAs).

created: Nov 19 2021
author: Henning Bathel

"""

logger = logging.getLogger(__name__)
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_DEVICES = ["R&S", "MokuGo"]


class fra_device:
    """Class containing FRA device specifications."""

    def __init__(self, devicefile):
        """Load provided FRA device file.

        Parameters
        ----------
        devicefile: str
            Provide name of device or devicefile.

        """
        if devicefile not in SUPPORTED_DEVICES:
            raise ValueError(
                f"Only the following devices are supported: {SUPPORTED_DEVICES}"
            )
        devicefile = os.path.join(PACKAGE_DIR, "devices", f"{devicefile}.json")
        with open(devicefile) as dev_file:
            device = json.load(dev_file)

        # populate fra with settings
        self.header = device["header"]
        self.frequency_label = device["frequency"]
        self.attenuation_label = device["magnitude"]
        self.attenuation_label_alt = device["magnitude_alt"]
        self.phase_label = device["phase"]
        self.is_gain = device["is_gain"]


def mag_phase_to_complex(Z_mag, phase):
    """Convert Bode form to complex numbers."""
    return Z_mag * np.exp(1j * np.deg2rad(phase))


def open_short_compensation(Z_meas, Z_open, Z_short):
    """
    Compensates the measured impedance with open and short reference measurements.

    please make sure the parameters stayed the same for all measurements

    Parameters
    ----------
    Z_meas: int or float or :class:`numpy.ndarray`
        measured impedance of the DUT
    Z_open, Z_short: int or float or :class:`numpy.ndarray`
        reference measurements with open / short circuit

    Returns
    -------
    input dependent,
        impedance of Z_dut compensated

    """
    Z_dut = (Z_meas - Z_short) / (1 - (Z_meas - Z_short) * (1 / Z_open))

    return Z_dut


def parallel(val_list):
    """Convenience function to calculate the value of a list
       of resistors in parallel (or capacitors in series).

    May be used to set R_device if a shunt resistor is used in parallel to device input

    Parameters
    ----------
    val_list: list of float
        values of the individual resistors in parallel

    Returns
    -------
    float
        apparent value of the parallel resistors
    """
    try:
        tmp = 0.0
        for e in val_list:
            tmp = tmp + 1 / e
        return 1 / tmp
    except ZeroDivisionError:
        logger.error("Elements must not be 0!")


def bode_to_impedance(frequency, attenuation, phase, R_device=1e6):
    r"""Bode diagram (Attenuation and phase) to Impedance calculator.

    Parameters
    ----------
    frequency: :class:`numpy.ndarray`
        Measurement frequencies
    attenuation: :class:`numpy.ndarray`
        Attenuation array
    phase: :class:`numpy.ndarray`
        Phase array
    R_device: float
        Input impedance of the FRA

    Returns
    -------
    :class:`numpy.ndarray`,
        Frequency as omega (2*pi*f)
    :class:`numpy.ndarray`, complex
        Impedance array

    Notes
    -----
    Given the expression for the magnitude of a voltage in dB is

    .. math::

        M_{db} = 20 \log{\frac{V_1}{V_{ref}}}

    we calculate the voltage ratio from the attenuation in dB as

    .. math::

        ratio_{voltage} = 10^{M_{dB} / 20}

    Then, the voltage divider rule results in the magnitude of the impedance as

    .. math::

        Z_{dut} = ratio * R_{shunt} - R_{shunt}

    """
    vratio = 10 ** (attenuation / 20)
    Z_dut = vratio * R_device - R_device

    omega = 2.0 * np.pi * frequency
    Z_dut_complex = mag_phase_to_complex(Z_dut, phase)

    return omega, Z_dut_complex


def wrap_phase(phase):
    """
    Wraps the phase to -90deg to 90deg
    TODO: maybe there is a python function for this.
    """
    while phase > 90:
        phase -= 180
    while phase < -90:
        phase += 180
    return phase


def read_bode_csv_dev(filename, devicesettings):
    """
    Special funtion to generate appr. format from provided device csv-files.

    Parameters
    ----------
    filename: string
        relative path to csv file
    devicesettings: dict
        information about device

    Returns
    -------
    :class:`numpy.ndarray`
        Frequency
    :class:`numpy.ndarray`
        Attenuation
    :class:`numpy.ndarray`
        Phase
    """
    data = pandas.read_csv(filename, header=devicesettings["header"])
    try:
        Phase = np.array(data[devicesettings["phase"]])
    except KeyError:
        logger.warning(
            "File is not UTF-8 encoded, try to read with ISO-8859-1 encoding."
        )
        data = pandas.read_csv(
            filename, header=devicesettings["header"], encoding="ISO-8859-1"
        )
        Phase = np.array(data[devicesettings["phase"]])

    Frequency = np.array(data[devicesettings["frequency"]])
    try:
        Attenuation = np.array(data[devicesettings["magnitude"]])
    except KeyError:
        logger.warning("Could not determine magnitude key. Tries magnitude-alt next.")
        try:
            Attenuation = np.array(data[devicesettings["magnitude_alt"]])
        except KeyError:
            logger.error("Could not determine the right key for magnitude.")

    if devicesettings["is_gain"]:
        Attenuation = -1.0 * Attenuation
        Phase = -1.0 * Phase

    return Frequency, Attenuation, Phase


def read_bode_csv(filename, devicename):
    """
    CSV to Bode Plot Parser.

    Parameters
    ----------
    filename: string
        relative path to csv file

    devicename: string
        specify json file which will load device parameters

    output: Bode Plot
    format: Frequency, Attenuation, Phase

    Notes
    -----
    This function was tested for a MokuGo (Liquid Instruments)
    and an Rohde & Schwarz oscilloscope RTB2004
    """
    if devicename not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Only the following devices are supported: {SUPPORTED_DEVICES}"
        )
    device_info = os.path.join(PACKAGE_DIR, "devices", f"{devicename}.json")
    print("Device info: ", device_info)
    with open(device_info) as dev_file:
        devicesettings = json.load(dev_file)
    return read_bode_csv_dev(filename, devicesettings)


def bode_csv_to_impedance(filename, devicename, R_device=1e6):
    """
    Convert Bode output (Attenuation and phase) to Impedance and save as CSV.

    Parameters
    ----------
    filename: string
        relative path to csv file with Bode info
    devicename: string
        "MokuGo" and "R&S" are supported (devices/xx.json), or provide own device
    R_device: float
        Input impedance of the device. Default is 1 MegOhm.

    Returns
    -------
    :class:`numpy.ndarray`,
        Frequency as omega (2*pi*f)
    :class:`numpy.ndarray`, complex
        Impedance array
    """
    frequency, attenuation, phase = read_bode_csv(filename, devicename)
    return bode_to_impedance(frequency, attenuation, phase, R_device=R_device)


def neisys_to_impedance(filename, header=2):
    """Convert neisys format to impedance."""
    data = pandas.read_csv(filename, header=header)
    frequencies = np.array(data["  Freq. [Hz]  "])
    Z_dut = np.array(data["  |Z| [ohm]  "])
    phase = np.array(data["  Phi [deg]"])

    omega = 2.0 * np.pi * frequencies
    Z = mag_phase_to_complex(Z_dut, phase)

    return omega, Z
