#    The ImpedanceFitter is a package to fit impedance spectra to
#    equivalent-circuit models using open-source software.
#
#    Copyright (C) 2021, 2023 Henning Bathel, henning.bathel2[AT]uni-rostock.de
#    Copyright (C) 2021, 2023 Julius Zimmermann, julius.zimmermann[AT]uni-rostock.de
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

import pandas
import json
import numpy as np
import logging
import os

"""
Collection of useful functions to get Impedance from Bode Diagram CSV files
measured with frequency response analysers (FRAs).

created: Nov 19 2021
author: Henning Bathel

"""

logger = logging.getLogger(__name__)
package_directory = os.path.dirname(os.path.abspath(__file__))


class fra_device():
    """Class containing FRA device specifications.

    """
    def __init__(self, devicefile):
        """Load provided FRA device file

        Parameters
        ----------
        devicefile: str
            Provide name of device or devicefile.

        """

        if devicefile in ["RuS", "MokuGo"]:
            devicefile = "{}/devices/{}.json".format(package_directory, devicefile)
        with open(f"{devicefile}", "r") as dev_file:
            device = json.load(dev_file)

        # populate fra with settings
        self.header = device["header"]
        self.frequency_label = device["frequency"]
        self.attenuation_label = device["magnitude"]
        self.attenuation_label_alt = device["magnitude_alt"]
        self.phase_label = device["phase"]
        self.is_gain = device["is_gain"]


def open_short_compensation(Z_meas, Z_open, Z_short):
    """
    compensates the measured impedance with open and short reference measurements

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
    """
    convenience function to calculate the value of a list of resistors in parallel (or capacitors in series)

    may be used to set R_device if a shunt resistor is used in parallel to device input

    Parameters
    ----------
    val_list: list of float
        values of the individual resistors in parallel
    Returns
    -------
    float,
        apparent value of the parallel resistors
    """
    try:
        tmp = 0.
        for e in val_list:
            tmp = tmp + 1 / e
        return 1 / tmp
    except ZeroDivisionError:
        logger.error("Elements must not be 0!")


def bode_to_impedance(frequency, attenuation, phase, R_device=1e6):
    """
    Bode diagram (Attenuation and phase) to Impedance calculator

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

    TBD: explain conversion formula
    """
    vratio = 10**(attenuation / 20)
    Z_dut = vratio * R_device - R_device
    R_dut = Z_dut * np.cos(np.radians(phase))
    X_dut = Z_dut * np.sin(np.radians(phase))
    omega = 2. * np.pi * frequency
    Z_dut_complex = R_dut + 1j * X_dut

    return omega, Z_dut_complex


def wrap_phase(phase):
    """
    wraps the phase to -90deg to 90deg
    TODO: maybe there is a python function for this
    """
    while(phase > 90):
        phase -= 180
    while(phase < -90):
        phase += 180
    return phase


def read_bode_csv_dev(filename, device):
    """
    special funtion to generate appr. format from provided device csv-files

    Parameters
    ----------
    filename: string
        relative path to csv file

    Returns
    ----------
    :class:`numpy.ndarray`
        Frequency
    :class:`numpy.ndarray`
        Attenuation
    :class:`numpy.ndarray`
        Phase
    """
    data = pandas.read_csv(filename, header=device["header"])

    Frequency = np.array(data[device["frequency"]])
    try:
        Attenuation = np.array(data[device["magnitude"]])
    except KeyError:
        logger.warning("Could not determine magnitude key. Tries magnitude-alt next.")
        try:
            Attenuation = np.array(data[device["magnitude_alt"]])
        except KeyError:
            logger.error("Could not determine the right key for magnitude.")

    Phase = np.array(data[device["phase"]])

    if device["is_gain"]:
        Attenuation = -1. * Attenuation
        Phase = -1. * Phase

    return Frequency, Attenuation, Phase


def read_bode_csv(filename, devicename):
    """
    CSV to Bode Plot Parser

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
    This function was tested for a MokuGo (Liquid Instruments) and an Rohde & Schwarz oscilloscope RTB2004
    """
    try:
        if(devicename in ["R&S", "MokuGo"]):
            devicename = f"{package_directory}/devices/{devicename}"
        with open(f"{devicename}.json", "r") as dev_file:
            device = json.load(dev_file)
        return read_bode_csv_dev(filename, device)
    except Exception as e:
        logger.error(f"Could not find the file: {e}")
        return 0, 0, 0


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
