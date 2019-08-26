#    The ImpedanceFitter is a package that provides means to fit impedance spectra to theoretical models using open-source software.
#
#    Copyright (C) 2018, 2019 Leonard Thiele, leonard.thiele[AT]uni-rostock.de
#    Copyright (C) 2018, 2019 Julius Zimmermann, julius.zimmermann[AT]uni-rostock.de
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
import os
import numpy as np

from lmfit import minimize, Parameters
import lmfit
import yaml
import pandas as pd
from .single_shell import plot_single_shell, single_shell_residual
from .double_shell import plot_double_shell, double_shell_residual
from .cole_cole import plot_cole_cole, cole_cole_residual, suspension_residual
from .utils import set_parameters_from_yaml, plot_dielectric_properties, load_constants_from_yaml
# TODO: throw error when data from file is calculated to be wrong(negative epsilon)?
# create logger
logger = logging.getLogger('logger')

"""
For the documentation, check ../latex/documentation_python.tex
"""


class Fitter(object):
    def __init__(self, directory=None, **kwargs):
        """
        provide directory if you run code from directory different to data directory
        provide kwargs as:
            model='SingleShell' OR 'DoubleShell'
            LogLevel='DEBUG' OR 'INFO'
            solvername='solver', choose among globalSolvers: basinhopping, differential_evolution; local Solvers: levenberg-marquardt, nelder-mead, least-squares
            inputformat='TXT' OR 'XLSX'
        possible extra values:
            excludeEnding=string for ending that should be ignored (if there are files with the same ending as the chosen inputformat)
            minimumFrequency=float
            maximumFrequency=float
        """
        self.model = kwargs['model']
        self.solvername = kwargs['solvername']
        self.inputformat = kwargs['inputformat']
        try:
            self.LogLevel = kwargs['LogLevel']  # log level: choose info for less verbose output
        except KeyError:
            self.LogLevel = 'INFO'
        try:
            self.minimumFrequency = kwargs['minimumFrequency']
        except KeyError:
            self.minimumFrequency = None
            pass  # does not matter if minimumFrequency or maximumFrequency are not specified
        try:
            self.maximumFrequency = kwargs['maximumFrequency']
        except KeyError:
            self.maximumFrequency = None
            pass  # does not matter if minimumFrequency or maximumFrequency are not specified
        try:
            self.excludeEnding = kwargs['excludeEnding']
        except KeyError:
            self.excludeEnding = "impossibleEndingLoL"
            pass
        try:
            self.solver_kwargs = kwargs['solver_kwargs']
        except KeyError:
            self.solver_kwargs = {}
            pass
        try:  # for debugging reasons use for instance only 1 data set
            self.data_sets = kwargs['data_sets']
        except KeyError:
            self.data_sets = None
        if directory is None:
            directory = os.getcwd()
        self.directory = directory + '/'
        logger.setLevel(self.LogLevel)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)

        # load constants
        self._load_constants()

    def _load_constants(self):
        self.constants = load_constants_from_yaml()
        self.constants['v1'] = (1. - self.constants['dm'] / self.constants['dm'])**3
        self.constants['v2'] = (self.constants['Rn'] / (self.constants['Rc'] - self.constants['dm']))**3
        self.constants['v3'] = (1. - self.constants['dn'] / self.constants['Rn'])**3

    def main(self, protocol=None):
        max_rows_tag = False
        self.protocol = protocol
        open('outfile.yaml', 'w')  # create output file
        for filename in os.listdir(self.directory):
            filename = os.fsdecode(filename)
            if filename.endswith(self.excludeEnding):
                logger.info("Skipped file {} due to excluded ending.".format(filename))
                break
            if self.inputformat == 'TXT' and filename.endswith(".TXT"):
                self.prepare_txt()
                if max_rows_tag is False:  # all files have the same number of datarows, this only has to be determined once
                    max_rows = self.get_max_rows(filename)
                    max_rows_tag = True
                self.readin_Data_from_file(filename, max_rows)
                self.fittedValues = self.process_data_from_file(filename)
                self.process_fitting_results(filename)
            elif self.inputformat == 'XLSX' and filename.endswith(".xlsx"):
                self.readin_Data_from_xlsx(filename)
                iters = len(self.zarray)
                if self.data_sets is not None:
                    iters = self.data_sets
                for i in range(iters):
                    self.Z = self.zarray[i]
                    self.fittedValues = self.process_data_from_file(filename)
                    self.process_fitting_results(filename + ' Row' + str(i))

    def get_max_rows(self, filename):
        '''
        determins the number of actual data rows
        '''
        txt_file = open(self.directory + filename)
        for num, line in enumerate(txt_file, 1):
            if self.trace_b in line:
                max_rows = num - self.skiprows_txt - self.skiprows_trace
                break
        txt_file.close()
        logger.debug('number of rows per trace is: ' + str(max_rows))
        return max_rows

    def readin_Data_from_xlsx(self, filename):
        """
        read in data that is structured like: frequency, real part of impedance, imaginary part of impedance
        there may be many different sets of impedance data
        """
        logger.info('going to process excel file: ' + self.directory + filename)
        EIS = pd.read_excel(self.directory + filename)
        values = EIS.values
        # filter values,  so that only  the ones in a certain range get taken.
        filteredvalues = np.empty((0, values.shape[1]))
        shift = [0.0, 0.0]
        if self.minimumFrequency is None:
            self.minimumFrequency = values[0, 0] - 10.  # to make checks work
            shift[0] = 10.
        if self.maximumFrequency is None:
            shift[1] = 10.
            self.maximumFrequency = values[-1, 0] + 10.  # to make checks work
        logger.debug("minimumFrequency is {}".format(self.minimumFrequency + shift[0]))
        logger.debug("maximumFrequency is {}".format(self.maximumFrequency - shift[1]))

        for i in range(values.shape[0]):
            if(values[i, 0] > self.minimumFrequency):
                if(values[i, 0] < self.maximumFrequency):
                    bufdict = values[i]
                    bufdict.shape = (1, bufdict.shape[0])  # change shape so it can be appended
                    filteredvalues = np.append(filteredvalues, bufdict, axis=0)
                else:
                    break
        values = filteredvalues

        f = values[:, 0]
        self.omega = 2. * np.pi * f
        # construct complex-valued array from float data
        self.zarray = np.zeros((np.int((values.shape[1] - 1) / 2), values.shape[0]), dtype=np.complex128)

        for i in range(np.int((values.shape[1] - 1) / 2)):  # will always be an int(always real and imag part)
            self.zarray[i] = values[:, (i * 2) + 1] + 1j * values[:, (i * 2) + 2]

    def process_fitting_results(self, filename):
        '''
        function writes output into yaml file, statistical analysis of the result
        '''
        values = dict(self.fittedValues.params.valuesdict())
        for key in values:
            values[key] = values.get(key).item()  # conversion into python native type
        data = {str(filename): values}
        outfile = open('outfile.yaml', 'a')  # append to the already existing file, or create it, if ther is no file
        yaml.dump(data, outfile)

    def prepare_txt(self):
        self.trace_b = 'TRACE: B'
        self.skiprows_txt = 21  # header rows inside the *.txt files
        self.skiprows_trace = 2  # line between traces blocks

    def readin_Data_from_file(self, filename, max_rows):
        """
        data from txt files get read in, returns array with omega and complex-valued Z
        """
        logger.debug('going to process  text file: ' + self.directory + filename)
        txt_file = open(self.directory + filename, 'r')
        try:
            fileDataArray = np.loadtxt(txt_file, delimiter='\t', skiprows=self.skiprows_txt, max_rows=max_rows)
        except ValueError as v:
            logger.error('Error in file ' + filename, v.arg)
        fileDataArray = np.array(fileDataArray)  # convert into numpy array
        filteredvalues = np.empty((0, fileDataArray.shape[1]))
        if self.minimumFrequency is None:
            self.minimumFrequency = fileDataArray[0, 0].astype(np.float)
            logger.debug("minimumFrequency is {}".format(self.minimumFrequency))
        if self.maximumFrequency is None:
            self.maximumFrequency = fileDataArray[-1, 0].astype(np.float)
            logger.debug("maximumFrequency is {}".format(self.maximumFrequency))
        for i in range(fileDataArray.shape[0]):
            if(fileDataArray[i, 0] > self.minimumFrequency and fileDataArray[i, 0] < self.maximumFrequency):
                bufdict = fileDataArray[i]
                bufdict.shape = (1, bufdict.shape[0])  # change shape so it can be appended
                filteredvalues = np.append(filteredvalues, bufdict, axis=0)
        fileDataArray = filteredvalues

        f = fileDataArray[:, 0].astype(np.float)
        self.omega = 2. * np.pi * f
        Z_real = fileDataArray[:, 1]
        Z_im = fileDataArray[:, 2]
        self.Z = Z_real + 1j * Z_im

    def process_data_from_file(self, filename):
        """
        fitting the data to the cole_cole_model first(compensation of the electrode polarization)
        """
        self.cole_cole_output = self.fit_to_cole_cole(self.omega, self.Z)
        if self.LogLevel == 'DEBUG':
            suspension_output = self.fit_to_suspension_model(self.omega, self.Z)
        if self.LogLevel == 'DEBUG':
            plot_cole_cole(self.omega, self.Z, self.cole_cole_output, filename, self.constants['c0'], self.constants['cf'])
            plot_dielectric_properties(self.omega, self.cole_cole_output, suspension_output)
        k_fit = self.cole_cole_output.params.valuesdict()['k']
        alpha_fit = self.cole_cole_output.params.valuesdict()['alpha']
        emed = self.cole_cole_output.params.valuesdict()['eh']
        # now we need to know k and alpha only
        # fit data to cell model
        if self.model == 'SingleShell':
            single_shell_output = self.fit_to_single_shell(self.omega, self.Z, k_fit, alpha_fit, emed)
            fit_output = single_shell_output
        else:
            double_shell_output = self.fit_to_double_shell(self.omega, self.Z, k_fit, alpha_fit, emed)
            fit_output = double_shell_output
        if self.LogLevel == 'DEBUG':
            if self.model == 'SingleShell':
                plot_single_shell(self.omega, self.Z, single_shell_output, filename, self.constants)
            else:
                plot_double_shell(self.omega, self.Z, double_shell_output, filename, self.constants)
        return fit_output

    def select_and_solve(self, solvername, residual, params, args):
        '''
        selects the needed solver and minimizes to the given residual
        '''
        self.solver_kwargs['method'] = solvername
        self.solver_kwargs['args'] = args
        return minimize(residual, params, **self.solver_kwargs)

    #######################################################################
    # suspension model section
    def fit_to_suspension_model(self, omega, input1):
        '''
        input 1 is the real part of epsilon
        '''
        logger.debug('#################################')
        logger.debug('fit data to pure suspension model')
        logger.debug('#################################')
        params = Parameters()
        params = set_parameters_from_yaml(params, 'suspension')
        result = None
        result = self.select_and_solve(self.solvername, suspension_residual, params, (omega, input1, self.constants['c0'], self.constants['cf']))
        logger.info(lmfit.fit_report(result))
        logger.info(result.params.pretty_print())
        return result

    ################################################################
    #  cole_cole_section
    def fit_to_cole_cole(self, omega, input1):
        # find more details in formula 6 and 7 of https://ieeexplore.ieee.org/document/6191683
        '''
        input 1 is either  Z in case of mode =='Matlab' or the real part of epsilon
        '''
        logger.debug('#################################################################')
        logger.debug('fit data to cole-cole model to account for electrode polarisation')
        logger.debug('#################################################################')
        params = Parameters()
        params = set_parameters_from_yaml(params, 'cole_cole')
        result = None
        iters = 1
        if self.protocol == "Iterative":
            iters = 2
        for i in range(iters):  # we have two iterations
            logger.info("###########\nFitting round {}\n###########".format(i + 1))
            if i == 1:
                # fix these two parameters, conductivity and eh
                params['conductivity'].set(vary=False, value=result.params['conductivity'].value)
                params['eh'].set(vary=False, value=result.params['eh'].value)
            result = self.select_and_solve(self.solvername, cole_cole_residual, params, args=(omega, input1, self.constants['c0'], self.constants['cf']))
            logger.info(lmfit.fit_report(result))
            logger.debug(result.params.pretty_print())
        return result

    #################################################################
    # single_shell_section
    def fit_to_single_shell(self, omega, input1, k_fit, alpha_fit, emed_fit):
        '''
        input is Z
        '''
        params = Parameters()
        params = set_parameters_from_yaml(params, 'single_shell')
        params.add('k', vary=False, value=k_fit)
        params.add('alpha', vary=False, value=alpha_fit)
        params.add('emed', vary=False, value=emed_fit)
        result = None
        iters = 1
        if self.protocol == "Iterative":
            iters = 2
        for i in range(iters):  # we have two iterations
            logger.info("###########\nFitting round {}\n###########".format(i + 1))
            if i == 1:
                # fix conductivity
                params['kmed'].set(vary=False, value=result.params.valuesdict()['kmed'])
            result = self.select_and_solve(self.solvername, single_shell_residual, params, args=(omega, input1, self.constants))
            logger.info(lmfit.fit_report(result))
            logger.info(result.message)
            logger.debug((result.params.pretty_print()))
        return result

    ################################################################
    # double_shell_section
    def fit_to_double_shell(self, omega, input1, k_fit, alpha_fit, emed_fit):
        '''
        input 1 is Z, k_fit, alpha_fit and emed_fit are the determined values in the cole-cole-fit
        '''
        logger.debug('##############################')
        logger.debug('fit data to double shell model')
        logger.debug('##############################')
        params = Parameters()
        params.add('k', vary=False, value=k_fit)
        params.add('alpha', vary=False, value=alpha_fit)
        params.add('emed', vary=False, value=emed_fit)
        params = set_parameters_from_yaml(params, 'double_shell')

        result = None
        iters = 1
        if self.protocol == "Iterative":
            iters = 4
        for i in range(iters):
            logger.info("###########\nFitting round {}\n###########".format(i + 1))
            if i == 1:
                params['kmed'].set(vary=False, value=result.params.valuesdict()['kmed'])
            if i == 2:
                params['km'].set(vary=False, value=result.params.valuesdict()['km'])
                params['em'].set(vary=False, value=result.params.valuesdict()['em'])
            if i == 3:
                params['kcp'].set(vary=False, value=result.params.valuesdict()['kcp'])
            result = self.select_and_solve(self.solvername, double_shell_residual, params, args=(omega, input1, self.constants))
            logger.info(lmfit.fit_report(result))
            logger.info(result.message)
            logger.debug(result.params.pretty_print())
        return result
