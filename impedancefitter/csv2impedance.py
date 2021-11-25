import math
import csv

import numpy as np
import matplotlib.pyplot as plt

"""
Collection of useful functions to get Impedance from Bode Diagram CSV files
   
created: Nov 19 2021
author: Henning Bathel

"""

#global R_device
R_device = 1e6

def setShuntResistor(value):
    global R_device
    R_device = value

def bode_to_impedance(bode):
    """
    Bode Diagram to Impedance calculator

    Parameters
    ----------
    bode: :class:`numpy.ndarray`
        bode plot with format [Frequency, Attenuation, Phase]

    Returns
    -------
    :class:`list`:
        :class:`numpy.ndarray`, 
            Frequency as omega (2*pi*f)
        :class:`numpy.ndarray`, complex
            Impedance array
    """
    R_dut = []
    X_dut = []
    omega = []
    Z_dut_complex = []

    for f,gain,phase in zip(bode[0],bode[1],bode[2]):
        vratio = math.sqrt(10**(gain / 10))
        Z_dut = vratio*R_device-R_device
        R_dut.append(Z_dut * math.cos(phase*math.pi/180))
        X_dut.append(Z_dut * math.sin(phase*math.pi/180))
        omega.append(2.*np.pi*f)
        Z_dut_complex.append(R_dut[len(R_dut)-1] + 1j* X_dut[len(X_dut)-1])


    return [np.array(omega), np.array((Z_dut_complex))]

def readBodeMoku(filename):
    """
    specialised function to generate appropriate format from MokuGo-csv files

    Parameters
    ----------
    filename: string
        relative path to csv file   

    Returns
    ----------
    :class:`numpy.ndarray`, 
        Bode plot with format [frequency, attenuation, phase]
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
            if(not any("%" in l for l in line)): #skip header 
                for k,item in enumerate(line):
                    if(k==0):
                        freq.append((float)(item))
                    elif(k==1):
                        gain_ch1.append((float)(item))
                    elif(k==2):
                        phase_ch1.append((float)(item))
                    elif(k==3):
                        gain_ch2.append((float)(item))
                    elif(k==4):
                        phase_ch2.append((float)(item))
                    elif(k==5):
                        gain_ratio.append((float)(item))
                    elif(k==6):
                        phase_ratio.append(wrapPhase((float)(item)))
                    else:
                        pass

    if(k<5):
        for gch1,gch2,pch1,pch2 in zip(gain_ch1,gain_ch2,phase_ch1,phase_ch2):
            gain_ratio.append(gch2 - gch1)
            phase_ratio.append(pch2 - pch1)
            
    return np.array([freq,gain_ratio,phase_ratio])


def readBodeRuS(filename):
    """
    special funtion to generate appr. format from Rohde and Schwarz csv-files

    Parameters
    ----------
    filename: string
        relative path to csv file
    Returns
    ----------
    :class:`numpy.ndarray`, 
        Bode plot with format [frequency, attenuation, phase]


    TODO: R&S saves gain, needs the basolute value maybe.
    """    
    Sample = []
    Frequency = []
    Gain = []
    Phase = []
    Amplitude = []

    with open(filename, 'r') as csvfile:
        bode_plot = csv.reader(csvfile)

        next(bode_plot)         #skip header 

        for line in bode_plot:
            for k,item in enumerate(line):
                if(k==0):
                    Sample.append((float)(item))
                elif(k==1):
                    Frequency.append((float)(item))
                elif(k==2):
                    Gain.append(abs((float)(item)))
                elif(k==3):
                    Phase.append((float)(item))
                elif(k==4):
                    Amplitude.append((float)(item))
                else:
                    pass

    return np.array([Frequency,Gain,Phase])

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

    divicename: string
        devicename to determine specialised functions

    output: Bode Plot 
    format: Frequency, Gain_ch1, Phase_ch1, Gain_ch2, Phase_ch2, optional: [Gain_Ratio, Phase_Ratio] 
    """

    if(devicename == "MokuGo"):
        bode = readBodeMoku(filename)
    elif(devicename == "R&S"):
        bode = readBodeRuS(filename)
    else:
        print("Could not determine the device. Please check spelling.")
    return bode

def csv_to_impedance(filename, devicename):
    return bode_to_impedance(readBode(filename,devicename))