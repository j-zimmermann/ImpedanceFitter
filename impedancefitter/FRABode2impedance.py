#    The ImpedanceFitter is a package to fit impedance spectra to
#    equivalent-circuit models using open-source software.
#
#    Copyright (C) 2021 Henning Bathel, henning.bathel2[AT]uni-rostock.de
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

import csv
import numpy as np

"""
Collection of useful functions to get Impedance from Bode Diagram CSV files
measured with frequency response analysers (FRAs).

created: Nov 19 2021
author: Henning Bathel

"""


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

    vratio = np.sqrt(10**(attenuation / 10))
    Z_dut = vratio * R_device - R_device
    R_dut = Z_dut * np.cos(phase * np.pi / 180)
    X_dut = Z_dut * np.sin(phase * np.pi / 180)
    omega = 2. * np.pi * frequency
    Z_dut_complex = R_dut + 1j * X_dut

    return omega, Z_dut_complex


def readBodeMoku(filename):
    """
    specialised function to generate appropriate format from MokuGo-csv files

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
    freq = []
    gain_ch1 = []
    phase_ch1 = []
    gain_ch2 = []
    phase_ch2 = []

    gain_ratio = []
    phase_ratio = []

    with open(filename, 'r') as csvfile:
        bode_plot = csv.reader(csvfile)

        for line in bode_plot:
            if(not any("%" in l for l in line)):  # skip header
                for k, item in enumerate(line):
                    if(k == 0):
                        freq.append((float)(item))
                    elif(k == 1):
                        gain_ch1.append((float)(item))
                    elif(k == 2):
                        phase_ch1.append((float)(item))
                    elif(k == 3):
                        gain_ch2.append((float)(item))
                    elif(k == 4):
                        phase_ch2.append((float)(item))
                    elif(k == 5):
                        gain_ratio.append((float)(item))
                    elif(k == 6):
                        phase_ratio.append(wrapPhase((float)(item)))
                    else:
                        pass

    if k < 5:
        for gch1, gch2, pch1, pch2 in zip(gain_ch1, gain_ch2, phase_ch1, phase_ch2):
            gain_ratio.append(gch2 - gch1)
            phase_ratio.append(pch2 - pch1)
    return np.array(freq), np.array(gain_ratio), np.array(phase_ratio)


def readBodeRuS(filename):
    """
    special funtion to generate appr. format from Rohde and Schwarz csv-files

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

    Notes
    -----
    TODO: R&S saves gain, needs the absolute value maybe.
    """
    Sample = []
    Frequency = []
    Gain = []
    Phase = []
    Amplitude = []

    with open(filename, 'r') as csvfile:
        bode_plot = csv.reader(csvfile)

        next(bode_plot)  # skip header
        for line in bode_plot:
            for k, item in enumerate(line):
                if(k == 0):
                    Sample.append((float)(item))
                elif(k == 1):
                    Frequency.append((float)(item))
                elif(k == 2):
                    Gain.append(abs((float)(item)))
                elif(k == 3):
                    Phase.append((float)(item))
                elif(k == 4):
                    Amplitude.append((float)(item))
                else:
                    pass

    return np.array(Frequency), np.array(Gain), np.array(Phase)


def wrapPhase(phase):
    """
    wraps the phase to -90deg to 90deg
    TODO: maybe there is a python function for this
    """
    while(phase > 90):
        phase -= 180
    while(phase < -90):
        phase += 180
    return phase


def readBode(filename, devicename):
    """
    CSV to Bode Plot Parser

    Parameters
    ----------
    filename: string
        relative path to csv file

    devicename: string
        devicename to determine specialised functions

    output: Bode Plot
    format: Frequency, Gain_ch1, Phase_ch1, Gain_ch2, Phase_ch2, optional: [Gain_Ratio, Phase_Ratio]

    Notes
    -----

    This function was tested for a MokuGo (Liquid Instruments) and an Rohde & Schwarz oscilloscope RTB2004
    """

    if(devicename == "MokuGo"):
        frequency, attenuation, phase = readBodeMoku(filename)
    elif(devicename == "R&S"):
        frequency, attenuation, phase = readBodeRuS(filename)
    else:
        raise RuntimeError("Could not determine the device. Please check spelling. Possible options are 'MokuGo' and 'R&S'.")
    return frequency, attenuation, phase


def bode_csv_to_impedance(filename, devicename, R_device=1e6):
    """
    Convert Bode output (Attenuation and phase) to Impedance and save as CSV.

    Parameters
    ----------
    filename: string
        relative path to csv file with Bode info
    devicename: string
        devicename, currently only MokuGo and R&S are supported
    R_device: float
        Input impedance of the device. Default is 1 MegOhm.

    Returns
    -------
    :class:`numpy.ndarray`,
        Frequency as omega (2*pi*f)
    :class:`numpy.ndarray`, complex
        Impedance array
    """
    frequency, attenuation, phase = readBode(filename, devicename)
    return bode_to_impedance(frequency, attenuation, phase, R_device=R_device)
