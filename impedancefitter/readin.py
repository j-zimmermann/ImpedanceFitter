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

import logging
import pandas as pd
import numpy as np
logger = logging.getLogger('impedancefitter-logger')


def readin_Data_from_collection(filepath, fileformat, minimumFrequency=None, maximumFrequency=None):
    """
    read in data collection from Excel or CSV file that is structured like: frequency, real part of impedance, imaginary part of impedance
    there may be many different sets of impedance data, i.e. there may be more columns with the real and the imaginary part.
    Then, the frequencies column must not be repeated.

    Parameters
    ----------

    filepath: string
        Provide the full filepath
    fileformat: string
        Provide fileformat. Possibilities are 'XLSX' and 'CSV'.
    minimumFrequency: optional
        Provide a minimum frequency. All values below this frequency will be ignored.
    maximumFrequency: optional
        Provide a maximum frequency. All values above this frequencies will be ignored.

    Returns
    -------

    omega: array of doubles
        frequency array
    zarray: array of complex
        contains collection of impedance spectra.
    """
    logger.info('going to process file: ' + filepath)
    if fileformat == 'XLSX':
        EIS = pd.read_excel(filepath)
    elif fileformat == 'CSV':
        EIS = pd.read_csv(filepath)
    else:
        raise NotImplementedError("File type not known")

    # sort frequencies to be on the safe side
    tmp = EIS.values
    values = tmp[tmp[:, 0].argsort()]

    # filter values,  so that only  the ones in a certain range get taken.
    filteredvalues = np.empty((0, values.shape[1]))
    shift = [0.0, 0.0]
    if minimumFrequency is None:
        minimumFrequency = values[0, 0] - 10.  # to make checks work
        shift[0] = 10.
    if maximumFrequency is None:
        shift[1] = 10.
        maximumFrequency = values[-1, 0] + 10.  # to make checks work
    logger.debug("minimumFrequency is {}".format(minimumFrequency + shift[0]))
    logger.debug("maximumFrequency is {}".format(maximumFrequency - shift[1]))

    for i in range(values.shape[0]):
        if(values[i, 0] > minimumFrequency):
            if(values[i, 0] < maximumFrequency):
                bufdict = values[i]
                bufdict.shape = (1, bufdict.shape[0])  # change shape so it can be appended
                filteredvalues = np.append(filteredvalues, bufdict, axis=0)
            else:
                break
    values = filteredvalues

    f = values[:, 0]
    omega = 2. * np.pi * f
    # construct complex-valued array from float data
    zarray = np.zeros((np.int((values.shape[1] - 1) / 2), values.shape[0]), dtype=np.complex128)

    for i in range(np.int((values.shape[1] - 1) / 2)):  # will always be an int(always real and imag part)
        zarray[i] = values[:, (i * 2) + 1] + 1j * values[:, (i * 2) + 2]
    return omega, zarray


def readin_Data_from_csv_E4980AL(filepath, minimumFrequency=None, maximumFrequency=None, current_threshold=None):
    """
    read in data that is structured like: frequency, real part of impedance, imaginary part of impedance, voltage, current

    .. note::
        There is always only one data set in a file.

    Parameters
    ----------

    filepath: string
        Provide the full filepath
    minimumFrequency: optional
        Provide a minimum frequency. All values below this frequency will be ignored.
    maximumFrequency: optional
        Provide a maximum frequency. All values above this frequencies will be ignored.
    current_threshold: optional
        Provides a current that the device had to pass through the sample. This threshold has
        to be met with 1% accuracy.

    Returns
    -------

    omega: array of doubles
        frequency array
    zarray: array of complex
        contains collection of impedance spectra.

    """
    logger.info('going to process csv file: ' + filepath)
    EIS = pd.read_csv(filepath)
    tmp = EIS.values
    values = tmp[tmp[:, 0].argsort()]  # need to sort frequencies
    # filter values,  so that only  the ones in a certain range get taken.
    filteredvalues = np.empty((0, 3))
    shift = [0.0, 0.0]
    if minimumFrequency is None:
        minimumFrequency = values[0, 0] - 10.  # to make checks work
        shift[0] = 10.
    if maximumFrequency is None:
        shift[1] = 10.
        maximumFrequency = values[-1, 0] + 10.  # to make checks work
    logger.debug("minimumFrequency is {}".format(minimumFrequency + shift[0]))
    logger.debug("maximumFrequency is {}".format(maximumFrequency - shift[1]))

    for i in range(values.shape[0]):
        if(values[i, 0] > minimumFrequency):
            if(values[i, 0] < maximumFrequency):
                bufdict = values[i][:3:]
                bufdict.shape = (1, bufdict.shape[0])  # change shape so it can be appended
                # in current-driven mode we need to check if the device was able to deliver the current
                # we assume 1% as the threshold
                if current_threshold is not None:
                    if not np.isclose(values[i][4], current_threshold, rtol=1e-2):
                        continue
                filteredvalues = np.append(filteredvalues, bufdict, axis=0)
            else:
                break
    values = filteredvalues

    f = values[:, 0]
    omega = 2. * np.pi * f
    # construct complex-valued array from float data
    zarray = np.zeros((np.int((values.shape[1] - 1) / 2), values.shape[0]), dtype=np.complex128)

    for i in range(np.int((values.shape[1] - 1) / 2)):  # will always be an int(always real and imag part)
        zarray[i] = values[:, (i * 2) + 1] + 1j * values[:, (i * 2) + 2]

    return omega, zarray


def _get_max_rows(filepath, trace_b, skiprows_txt, skiprows_trace):
    '''
    determines the number of actual data rows in TXT files.
    '''

    txt_file = open(filepath)
    for num, line in enumerate(txt_file, 1):
        if trace_b in line:
            max_rows = num - skiprows_txt - skiprows_trace
            break
    txt_file.close()
    logger.debug('number of rows per trace is: ' + str(max_rows))
    return max_rows


def readin_Data_from_file(filepath, skiprows_txt, skiprows_trace, trace_b, minimumFrequency=None, maximumFrequency=None):
    """
    Data from txt files get reads in, returns array with omega and complex-valued impedance Z.
    The TXT files may contain two traces; only one of them is read in.

    Parameters
    ----------

    filepath: string
        Provide the full filepath
    skiprows_txt: int
        header rows inside the *.txt file
    skiprows_trace: int
        lines between traces blocks
    trace_b: string
        Flag for beginning of second trace in data.
    fileformat: string
        Provide fileformat. Possibilities are 'XLSX' and 'CSV'.
    minimumFrequency: optional
        Provide a minimum frequency. All values below this frequency will be ignored.
    maximumFrequency: optional
        Provide a maximum frequency. All values above this frequencies will be ignored.

    Returns
    -------

    omega: array of doubles
        frequency array
    Z: array of complex
        contains impedance spectrum

    """
    logger.debug('going to process  text file: ' + filepath)
    max_rows = _get_max_rows(filepath, trace_b, skiprows_txt, skiprows_trace)
    txt_file = open(filepath, 'r')
    try:
        fileDataArray = np.loadtxt(txt_file, delimiter='\t', skiprows=skiprows_txt, max_rows=max_rows)
    except ValueError as v:
        logger.error('Error in file ' + filepath, v.arg)
    fileDataArray = np.array(fileDataArray)  # convert into numpy array
    filteredvalues = np.empty((0, fileDataArray.shape[1]))
    if minimumFrequency is None:
        minimumFrequency = fileDataArray[0, 0].astype(np.float)
        logger.debug("minimumFrequency is {}".format(minimumFrequency))
    if maximumFrequency is None:
        maximumFrequency = fileDataArray[-1, 0].astype(np.float)
        logger.debug("maximumFrequency is {}".format(maximumFrequency))
    for i in range(fileDataArray.shape[0]):
        if(fileDataArray[i, 0] > minimumFrequency and fileDataArray[i, 0] < maximumFrequency):
            bufdict = fileDataArray[i]
            bufdict.shape = (1, bufdict.shape[0])  # change shape so it can be appended
            filteredvalues = np.append(filteredvalues, bufdict, axis=0)
    fileDataArray = filteredvalues

    f = fileDataArray[:, 0].astype(np.float)
    omega = 2. * np.pi * f
    Z_real = fileDataArray[:, 1]
    Z_im = fileDataArray[:, 2]
    Z = Z_real + 1j * Z_im
    return omega, Z
