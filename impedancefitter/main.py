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
import matplotlib.pyplot as plt
import openturns as ot
from openturns.viewer import View

from lmfit import minimize, Parameters
import lmfit
import yaml
import pandas as pd
from .single_shell import plot_single_shell, single_shell_residual
from .double_shell import plot_double_shell, double_shell_residual
from .cole_cole import plot_cole_cole, cole_cole_residual, suspension_residual
from .utils import set_parameters_from_yaml, plot_dielectric_properties
# TODO: throw error when data from file is calculated to be wrong(negative epsilon)?
# create logger
logger = logging.getLogger('logger')

if os.path.isfile('./constants.py'):
    print("Using constants specified in directory.")
    import importlib.util
    spec = importlib.util.spec_from_file_location("module.name", os.getcwd() + "/constants.py")
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)

else:
    print("Using default constants.")
    import impedancefitter.constants as constants
"""
For the documentation, check ../latex/documentation_python.tex
"""


class Fitter(object):
    def __init__(self, **kwargs):
        """
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
        self.LogLevel = kwargs['LogLevel']  # log level: choose info for less verbose output
        self.solvername = kwargs['solvername']
        self.inputformat = kwargs['inputformat']
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

        self.directory = os.getcwd() + '/'
        logger.setLevel(self.LogLevel)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)

    def main(self, protocol=None):
        max_rows_tag = False
        self.protocol = protocol
        open(self.directory + 'outfile.yaml', 'w')  # create output file
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
                dataArray = self.readin_Data_from_file(filename, max_rows)
                fittedValues = self.process_data_from_file(dataArray, filename)
                # dataArray in the form [omega, Z, Y, epsilon, k]
                self.process_fitting_results(fittedValues, filename)
            elif self.inputformat == 'XLSX' and filename.endswith(".xlsx"):
                dataArray = self.readin_Data_from_xlsx(filename)
                iters = dataArray[1].shape[0]
                if self.data_sets is not None:
                    iters = self.data_sets
                for i in range(iters):
                    fittedValues = self.process_data_from_file([dataArray[0], dataArray[1][i]], filename)
                    # dataArray in the form [omega, Z, Y, epsilon, k]
                    self.process_fitting_results(fittedValues, filename + ' Row' + str(i))

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

    def prepare_post_processing(self):
        '''
        draws pdf of all calculated parameters
        '''
        file = open(self.directory + 'outfile.yaml', 'r')
        data = yaml.safe_load(file)
        alphalist, emlist, klist, kmlist, kcplist, kmedlist, emedlist, enelist, knelist, knplist = ([] for i in range(10))
        for key in data:
            alphalist.append([data[key]['alpha']])
            emlist.append([data[key]['em']])
            klist.append([data[key]['k']])
            kmlist.append([data[key]['km']])
            kcplist.append([data[key]['kcp']])
            kmedlist.append([data[key]['kmed']])
            emedlist.append([data[key]['emed']])
            if(self.model == 'DoubleShell'):
                enelist.append([data[key]['ene']])
                knelist.append([data[key]['kne']])
                knplist.append([data[key]['knp']])
        # write data into dict
        self.sampledict = {}
        self.sampledict['alpha'] = ot.Sample(np.array(alphalist))
        self.sampledict['em'] = ot.Sample(np.array(emlist))
        self.sampledict['k'] = ot.Sample(np.array(klist))
        self.sampledict['km'] = ot.Sample(np.array(kmlist))
        self.sampledict['kcp'] = ot.Sample(np.array(kcplist))
        self.sampledict['kmed'] = ot.Sample(np.array(kmedlist))
        self.sampledict['emed'] = ot.Sample(np.array(emedlist))
        if(self.model == 'DoubleShell'):
            self.sampledict['ene'] = ot.Sample(np.array(enelist))
            self.sampledict['kne'] = ot.Sample(np.array(knelist))
            self.sampledict['knp'] = ot.Sample(np.array(knplist))

    def plot_histograms(self):
        """
        fails if values are too close to each other
        """
        if(self.model == 'SingleShell'):
            fig, ax = plt.subplots(nrows=2, ncols=3)
        else:
            fig, ax = plt.subplots(nrows=3, ncols=3)
        r = 0
        c = 0
        for key in self.sampledict:
            graph = ot.HistogramFactory().build(self.sampledict[key]).drawPDF()
            graph.setTitle("Histogram for variables")
            graph.setXTitle(key)
            View(graph, axes=[ax[r, c]], plot_kwargs={'label': key, 'c': 'black'})
            kernel = ot.KernelSmoothing()
            graph_k = kernel.build(self.sampledict[key])
            graph_k = graph_k.drawPDF()
            graph_k.setTitle("Histogram for variables")
            graph_k.setXTitle(key)
            View(graph_k, axes=[ax[r, c]], plot_kwargs={'label': key})
            # jump to next ax object or next row
            c += 1
            if(c == 3):
                c = 0
                r += 1
        plt.tight_layout()
        plt.show()

    def fit_to_normal_distribution(self, parameter):
        sample = self.sampledict[parameter]
        distribution = ot.NormalFactory().build(sample)
        print(distribution)
        # Draw QQ plot to check fitted distribution
        QQ_plot = ot.VisualTest.DrawQQplot(sample, distribution)
        View(QQ_plot).show()
        return distribution

    def fit_to_histogram_distribution(self, parameter):
        sample = self.sampledict[parameter]
        distribution = ot.HistogramFactory().build(sample)
        print(distribution)
        # Draw QQ plot to check fitted distribution
        QQ_plot = ot.VisualTest.DrawQQplot(sample, distribution)
        View(QQ_plot).show()
        return distribution

    def best_model_kolmogorov(self, parameter):
        """
        suitable for small samples
        """
        sample = self.sampledict[parameter]
        tested_distributions = [ot.NormalFactory(), ot.UniformFactory()]
        best_model, best_result = ot.FittingTest.BestModelKolmogorov(sample, tested_distributions)
        logger.info("Best model:")
        logger.info(best_model)
        if self.LogLevel == 'DEBUG':
            logger.debug("QQ Plot for best model:")
            QQ_plot = ot.VisualTest.DrawQQplot(sample, best_model)
            View(QQ_plot).show()
        return best_model

    def best_model_chisquared(self, parameter):
        sample = self.sampledict[parameter]
        tested_distributions = [ot.ExponentialFactory(), ot.NormalFactory(), ot.BetaFactory(), ot.UniformFactory(), ot.TruncatedNormalFactory()]
        best_model, best_result = ot.FittingTest.BestModelChiSquared(sample, tested_distributions)
        logger.info("Best model:")
        logger.info(best_model)
        return best_model

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
        omega = 2. * np.pi * f
        # construct complex-valued array from float data
        zarray = np.zeros((np.int((values.shape[1] - 1) / 2), values.shape[0]), dtype=np.complex128)

        for i in range(np.int((values.shape[1] - 1) / 2)):  # will always be an int(always real and imag part)
            zarray[i] = values[:, (i * 2) + 1] + 1j * values[:, (i * 2) + 2]
        return(omega, zarray)

    def process_fitting_results(self, fittedValues, filename):
        '''
        function writes output into yaml file, statistical analysis of the result
        '''
        values = dict(fittedValues.params.valuesdict())
        for key in values:
            values[key] = values.get(key).item()  # conversion into python native type
        data = {str(filename): values}
        outfile = open(self.directory + 'outfile.yaml', 'a')  # append to the already existing file, or create it, if ther is no file
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
        omega = 2. * np.pi * f
        Z_real = fileDataArray[:, 1]
        Z_im = fileDataArray[:, 2]
        Z = Z_real + 1j * Z_im
        return([omega, Z])

    def process_data_from_file(self, dataArray, filename):
        """
        fitting the data to the cole_cole_model first(compensation of the electrode polarization)
        """
        self.cole_cole_output = self.fit_to_cole_cole(dataArray[0], dataArray[1])
        if self.LogLevel == 'DEBUG':
            suspension_output = self.fit_to_suspension_model(dataArray[0], dataArray[1])
        if self.LogLevel == 'DEBUG':
            plot_cole_cole(dataArray[0], dataArray[1], self.cole_cole_output, filename)
            plot_dielectric_properties(dataArray[0], self.cole_cole_output, suspension_output)
        k_fit = self.cole_cole_output.params.valuesdict()['k']
        alpha_fit = self.cole_cole_output.params.valuesdict()['alpha']
        emed = self.cole_cole_output.params.valuesdict()['eh']
        # now we need to know k and alpha only
        # fit data to cell model
        if self.model == 'SingleShell':
            single_shell_output = self.fit_to_single_shell(dataArray[0], dataArray[1], k_fit, alpha_fit, emed)
            fit_output = single_shell_output
        else:
            double_shell_output = self.fit_to_double_shell(dataArray[0], dataArray[1], k_fit, alpha_fit, emed)
            fit_output = double_shell_output
        if self.LogLevel == 'DEBUG':
            if self.model == 'SingleShell':
                plot_single_shell(dataArray[0], dataArray[1], single_shell_output, filename)
            else:
                plot_double_shell(dataArray[0], dataArray[1], double_shell_output, filename)
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
        result = self.select_and_solve(self.solvername, suspension_residual, params, (omega, input1))
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
            result = self.select_and_solve(self.solvername, cole_cole_residual, params, args=(omega, input1))
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
        result = self.select_and_solve(self.solvername, single_shell_residual, params, args=(omega, input1))
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
            result = self.select_and_solve(self.solvername, double_shell_residual, params, args=(omega, input1))
            logger.info(lmfit.fit_report(result))
            logger.info(result.message)
            logger.debug(result.params.pretty_print())
        return result
