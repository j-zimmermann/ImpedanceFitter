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

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def readin_Data_from_collection(filepath, fileformat, delimiter=None,
                                minimumFrequency=None, maximumFrequency=None):
    """read in data collection from Excel or CSV file.

    The file is structured like:
    frequency, real part of impedance, imaginary part of impedance.
    There may be many different sets of impedance data,
    i.e. there may be more columns with the real and the imaginary part.
    Then, the frequencies column must not be repeated.

    Parameters
    ----------

    filepath: string
        Provide the full filepath
    fileformat: string
        Provide fileformat. Possibilities are 'XLSX' and 'CSV'.
    minimumFrequency: float, optional
        Provide a minimum frequency. All values below this frequency will be ignored.
    maximumFrequency: float, optional
        Provide a maximum frequency. All values above this frequencies will be ignored.

    Returns
    -------

    omega: :class:`numpy.ndarray`
        frequency array
    zarray:  :class:`numpy.ndarray`
        Contains collection of impedance spectra. Has shape (number of spectra, number of frequencies).
    """
    logger.info('going to process file: ' + filepath)
    if fileformat == 'XLSX':
        if delimiter is not None:
            logger.warning("You provided a delimiter for an XLSX file but it has no effect.")
        EIS = pd.read_excel(filepath)
    elif fileformat == 'CSV':
        EIS = pd.read_csv(filepath, delimiter=delimiter)
    else:
        raise NotImplementedError("File type not known")

    # sort frequencies to be on the safe side
    tmp = EIS.values
    values = tmp[tmp[:, 0].argsort()]

    # filter values,  so that only  the ones in a certain range get taken.
    filteredvalues = np.empty((0, values.shape[1]))
    if minimumFrequency is None:
        minimumFrequency = values[0, 0]
    if maximumFrequency is None:
        maximumFrequency = values[-1, 0]
    logger.info("minimumFrequency is {}".format(minimumFrequency))
    logger.info("maximumFrequency is {}".format(maximumFrequency))

    for i in range(values.shape[0]):
        if np.greater_equal(values[i, 0], minimumFrequency):
            if np.less_equal(values[i, 0], maximumFrequency):
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


def readin_Data_from_csv_E4980AL(filepath, minimumFrequency=None, maximumFrequency=None, current_threshold=None,
                                 voltage_threshold=None, tolerance=1e-2):
    """Read in data from E4980AL-LCR meter.

    Read in data that is structured like:
    frequency, real part of impedance, imaginary part of impedance, voltage, current

    .. note::
        There is always only one data set in a file.

    Parameters
    ----------

    filepath: string
        Provide the full filepath
    minimumFrequency: float, optional
        Provide a minimum frequency. All values below this frequency will be ignored.
    maximumFrequency: float, optional
        Provide a maximum frequency. All values above this frequencies will be ignored.
    current_threshold: float, optional
        Provides a current that the device had to pass through the sample. This threshold has
        to be met with the accuracy given by `tolerance`.
    voltage_threshold: float, optional
        Provides a voltage that the device had to apply. This threshold has
        to be met with the accuracy given by `tolerance`.
    tolerance: float, optional
        `tolerance` level for voltage and/or current. Default is 1e-2, which refers to 1% accuracy.


    Returns
    -------

    omega: :class:`numpy.ndarray`
        frequency array
    zarray:  :class:`numpy.ndarray`
        Contains collection of impedance spectra. Has shape (1, number of frequencies).

    """
    logger.info('going to process csv file: ' + filepath)
    EIS = pd.read_csv(filepath)
    tmp = EIS.values
    values = tmp[tmp[:, 0].argsort()]  # need to sort frequencies
    # filter values,  so that only  the ones in a certain range get taken.
    filteredvalues = np.empty((0, 3))
    if minimumFrequency is None:
        minimumFrequency = values[0, 0]
    if maximumFrequency is None:
        maximumFrequency = values[-1, 0]

    for i in range(values.shape[0]):
        if np.greater_equal(values[i, 0], minimumFrequency):
            if np.less_equal(values[i, 0], maximumFrequency):
                bufdict = values[i][:3:]
                bufdict.shape = (1, bufdict.shape[0])  # change shape so it can be appended
                # in current-driven mode we need to check if the device was able to deliver the current
                # we assume 1% as the threshold
                if current_threshold is not None:
                    if not np.isclose(values[i][4], current_threshold, rtol=tolerance):
                        continue
                if voltage_threshold is not None:
                    if not np.isclose(values[i][3], voltage_threshold, rtol=tolerance):
                        continue
                filteredvalues = np.append(filteredvalues, bufdict, axis=0)
            else:
                break
    values = filteredvalues

    f = values[:, 0]
    omega = 2. * np.pi * f

    if f.size > 0:
        logger.info("minimumFrequency is {}".format(f.min()))
        logger.info("maximumFrequency is {}".format(f.max()))

    # construct complex-valued array from float data
    zarray = np.zeros((np.int((values.shape[1] - 1) / 2), values.shape[0]), dtype=np.complex128)

    for i in range(np.int((values.shape[1] - 1) / 2)):  # will always be an int(always real and imag part)
        zarray[i] = values[:, (i * 2) + 1] + 1j * values[:, (i * 2) + 2]

    return omega, zarray


def _get_max_rows(filepath, trace_b, skiprows_txt, skiprows_trace):
    '''
    determines the number of actual data rows in TXT files.
    '''

    max_rows = -1
    txt_file = open(filepath)
    for num, line in enumerate(txt_file, 1):
        if trace_b in line:
            max_rows = num - skiprows_txt - skiprows_trace
            break
    txt_file.close()
    if max_rows < 0:
        raise RuntimeError("Could not process TXT file, second trace could not be found")
    logger.debug('number of rows per trace is: ' + str(max_rows))
    return max_rows


def readin_Data_from_TXT_file(filepath, skiprows_txt, skiprows_trace=None,
                              trace_b=None, delimiter="\t", minimumFrequency=None, maximumFrequency=None):
    """Read in data from TXT file.

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
    delimiter: string, optional
        Choose the delimiter. Default is tab, i.e. tab-delimited data is read in.
    minimumFrequency: optional
        Provide a minimum frequency. All values below this frequency will be ignored.
    maximumFrequency: optional
        Provide a maximum frequency. All values above this frequencies will be ignored.

    Returns
    -------

    omega: :class:`numpy.ndarray`
        frequency array
    zarray:  :class:`numpy.ndarray`
        Contains collection of impedance spectra. Has shape (1, number of frequencies).

    """
    logger.info('going to process  text file: ' + filepath)
    max_rows = None  # numpy default
    if trace_b is not None:
        max_rows = _get_max_rows(filepath, trace_b, skiprows_txt, skiprows_trace)
    txt_file = open(filepath, 'r')
    try:
        fileDataArray = np.loadtxt(txt_file, delimiter=delimiter,
                                   skiprows=skiprows_txt, max_rows=max_rows)
    except ValueError as v:
        logger.error('Error in file {}.\n {}'.format(filepath, v.args))
        raise
    filteredvalues = np.empty((0, fileDataArray.shape[1]))
    if minimumFrequency is None:
        minimumFrequency = fileDataArray[0, 0].astype(np.float)
        logger.info("minimumFrequency is {}".format(minimumFrequency))
    if maximumFrequency is None:
        maximumFrequency = fileDataArray[-1, 0].astype(np.float)
        logger.info("maximumFrequency is {}".format(maximumFrequency))
    for i in range(fileDataArray.shape[0]):
        if (np.greater_equal(fileDataArray[i, 0], minimumFrequency)
               and np.less_equal(fileDataArray[i, 0], maximumFrequency)):
            bufdict = fileDataArray[i]
            bufdict.shape = (1, bufdict.shape[0])  # change shape so it can be appended
            filteredvalues = np.append(filteredvalues, bufdict, axis=0)
    fileDataArray = filteredvalues

    f = fileDataArray[:, 0].astype(np.float)
    omega = 2. * np.pi * f
    Z_real = fileDataArray[:, 1]
    Z_im = fileDataArray[:, 2]
    Z = Z_real + 1j * Z_im
    return omega, np.array([Z])


def readin_Data_from_dta(filepath, minimumFrequency=None, maximumFrequency=None):
    """Read in data from DTA data (Gamry).

    .. note::
        There is always only one data set in a file.

    Parameters
    ----------

    filepath: string
        Provide the full filepath
    minimumFrequency: float, optional
        Provide a minimum frequency. All values below this frequency will be ignored.
    maximumFrequency: float, optional
        Provide a maximum frequency. All values above this frequencies will be ignored.


    Returns
    -------

    omega: :class:`numpy.ndarray`
        frequency array
    zarray:  :class:`numpy.ndarray`
        Contains collection of impedance spectra. Has shape (1, number of frequencies).

    """
    logger.info('going to process DTA file: ' + filepath)
    with open(filepath, encoding='utf-8', errors='ignore') as w:
        lines = w.readlines()
    index = lines.index("ZCURVE\tTABLE\n")
    freq = []
    Zreal = []
    Zimag = []
    for line in lines[index + 3::]:
        data = line.split("\t")
        assert len(data) == 12, "Line {} does not contain enough data!"
        freq.append(float(data[3].replace(",", ".")))
        Zreal.append(float(data[4].replace(",", ".")))
        Zimag.append(float(data[5].replace(",", ".")))

    tmp = np.column_stack([freq, Zreal, Zimag])
    values = tmp[tmp[:, 0].argsort()]  # need to sort frequencies
    # filter values,  so that only  the ones in a certain range get taken.
    filteredvalues = np.empty((0, 3))
    if minimumFrequency is None:
        minimumFrequency = values[0, 0]
    if maximumFrequency is None:
        maximumFrequency = values[-1, 0]

    for i in range(values.shape[0]):
        if np.greater_equal(values[i, 0], minimumFrequency):
            if np.less_equal(values[i, 0], maximumFrequency):
                bufdict = values[i][:3:]
                bufdict.shape = (1, bufdict.shape[0])  # change shape so it can be appended
                filteredvalues = np.append(filteredvalues, bufdict, axis=0)
            else:
                break
    values = filteredvalues

    f = values[:, 0]
    omega = 2. * np.pi * f

    if f.size > 0:
        logger.info("minimumFrequency is {}".format(f.min()))
        logger.info("maximumFrequency is {}".format(f.max()))

    # construct complex-valued array from float data
    zarray = np.zeros((np.int((values.shape[1] - 1) / 2), values.shape[0]), dtype=np.complex128)

    for i in range(np.int((values.shape[1] - 1) / 2)):  # will always be an int(always real and imag part)
        zarray[i] = values[:, (i * 2) + 1] + 1j * values[:, (i * 2) + 2]

    return omega, zarray
