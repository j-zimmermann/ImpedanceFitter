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

import csv
from lmfit import minimize, Parameters
import lmfit
import yaml
import pandas as pd

if os.path.isfile('./constants.py'):
    import importlib.util
    spec = importlib.util.spec_from_file_location("module.name", os.getcwd() + "/constants.py")
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)

else:
    import impedancefitter.constants as constants
"""
For the documentation, check ../latex/documentation_python.tex
"""
# TODO: throw error when data from file is calculated to be wrong(negative epsilon)?
# create logger
logger = logging.getLogger('logger')


class Fitter(object):
    def __init__(self, **kwargs):
        """
        provide kwargs as:
            mode=None
            model='SingleShell' OR 'DoubleShell'
            LogLevel='DEBUG' OR 'INFO'
            solvername='solver', choose among globalSolvers: basinhopping, differential_evolution; local Solvers: levenberg-marquardt, nelder-mead, least-squares
            inputformat='TXT' OR 'XLSX'
        """
        self.mode = kwargs['mode']
        self.model = kwargs['model']
        self.LogLevel = kwargs['LogLevel']  # log level: choose info for less verbose output
        self.solvername = kwargs['solvername']
        self.inputformat = kwargs['inputformat']
        self.minimumFrequency = kwargs['minimumFrequency']
        self.maximumFrequency = kwargs['maximumFrequency']
        self.directory = os.getcwd() + '/'
        logger.setLevel(self.LogLevel)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)

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

    def main(self):
        max_rows_tag = False
        open(self.directory + 'outfile.yaml', 'w')  # create output file
        for filename in os.listdir(self.directory):
            filename = os.fsdecode(filename)
            if filename.endswith(".TXT"):
                self.prepare_txt()
                if max_rows_tag is False:  # all files have the same number of datarows, this only has to be determined once
                    max_rows = self.get_max_rows(filename)
                    max_rows_tag = True
                dataArray = self.readin_Data_from_file(filename, max_rows)
                fittedValues = self.process_data_from_file(dataArray, filename)
                # dataArray in the form [omega, Z, Y, epsilon, k]
                self.process_fitting_results(fittedValues, filename)
            if filename.endswith(".xlsx"):
                dataArray = self.readin_Data_from_xlxs(filename)
                for i in range(dataArray[1].shape[0]):
                    fittedValues = self.process_data_from_file([dataArray[0], dataArray[1][i]], filename)
                    # dataArray in the form [omega, Z, Y, epsilon, k]
                    self.process_fitting_results(fittedValues, filename + ' Row' + str(i))
        self.post_process_fitted_values()

    def post_process_fitted_values(self):
        '''
        draws pdf of all calculated parameters
        '''
        file = open(self.directory + 'outfile.yaml', 'r')
        data = yaml.load(file)
        alphalist, emlist, klist, kmlist, kcplist, kmedlist, enelist, knelist, knplist = ([] for i in range(9))
        for key in data:
            alphalist.append([data[key]['alpha']])
            emlist.append([data[key]['em']])
            klist.append([data[key]['k']])
            kmlist.append([data[key]['km']])
            kcplist.append([data[key]['kcp']])
            kmedlist.append([data[key]['kmed']])
            if(self.model == 'DoubleShell'):
                enelist.append([data[key]['ene']])
                knelist.append([data[key]['kne']])
                knplist.append([data[key]['knp']])
        # write data into dict
        sampledict = {}
        sampledict['alpha'] = ot.Sample(np.array(alphalist))
        sampledict['em'] = ot.Sample(np.array(emlist))
        sampledict['k'] = ot.Sample(np.array(klist))
        sampledict['km'] = ot.Sample(np.array(kmlist))
        sampledict['kcp'] = ot.Sample(np.array(kcplist))
        sampledict['kmed'] = ot.Sample(np.array(kmedlist))
        if(self.model == 'DoubleShell'):
            sampledict['ene'] = ot.Sample(np.array(enelist))
            sampledict['kne'] = ot.Sample(np.array(knelist))
            sampledict['knp'] = ot.Sample(np.array(knplist))
        if(self.model == 'SingleShell'):
            fig, ax = plt.subplots(nrows=2, ncols=3)
        else:
            fig, ax = plt.subplots(nrows=3, ncols=3)
        r = 0
        c = 0
        for key in sampledict:
            graph = ot.HistogramFactory().build(sampledict[key]).drawPDF()  # drawpdf doesnt work for km, values are too equal
            graph.setTitle(self.directory)  # TODO: crashes here with out of memory exception(single-shell)
            graph.setXTitle(key)
            View(graph, axes=[ax[r, c]], plot_kwargs={'label': key, 'c': 'black'})
            # graph = ot.HistogramFactory().build(sampledict[key])#crashed here
            kernel = ot.KernelSmoothing()
            graph_k = kernel.build(sampledict[key])
            # graph = ot.KernelSmoothing().computeSilvermanBandwidth(sampledict[key])
            graph_k = graph_k.drawPDF()
            graph_k.setTitle(self.directory + 'kernel')
            graph_k.setXTitle(key)
            View(graph_k, axes=[ax[r, c]], plot_kwargs={'label': key})
            # jump to next ax object or next row
            c += 1
            if(c == 3):
                c = 0
                r += 1
    #    view = View(graph)
    #    view.save('test.png')
        plt.show()

    def readin_Data_from_xlxs(self, filename):
        EIS = pd.read_excel(self.directory + filename)
        values = EIS.values
        # filter values,  so that only  the ones in a certain range get taken.
        filteredvalues = np.empty(0, values.shape[1])
        for i in range(values.shape[0]):
            if(values[i, 0] > self.minimumFrequency and values[i, 0] < self.maximumFrequency):
                bufdict = values[i]
                bufdict.shape = (1, bufdict.shape[0])  # change shape so it can be appended
                filteredvalues = np.append(filteredvalues, bufdict, axis=0)
        values = filteredvalues

        f = values[:, 0]
        omega = 2. * np.pi * f
        zarray = np.zeros((np.int((values.shape[1] - 1) / 2), values.shape[0]), dtype=np.complex128)
        yarray = np.zeros((np.int((values.shape[1] - 1) / 2), values.shape[0]), dtype=np.complex128)
        epsilon = np.zeros((np.int((values.shape[1] - 1) / 2), values.shape[0]), dtype=np.complex128)
        k = np.zeros((np.int((values.shape[1] - 1) / 2), values.shape[0]), dtype=np.float64)

        for i in range(np.int((values.shape[1] - 1) / 2)):  # will always be an int(always real and imag part)
            zarray[i] = values[:, (i * 2) + 1] + 1j * values[:, (i * 2) + 2]
            if(self.mode == 'Matlab'):
                yarray[i] = 1. / zarray[i]
                epsilon[i] = (yarray[i] - 1j * omega * constants.cf) / (1j * omega * constants.c0)
                k[i] = -epsilon[i].imag * omega * constants.e0
        if(self.mode == 'Matlab'):
            return(omega, zarray, yarray, epsilon, k)
        return(omega, zarray)

    def process_fitting_results(self, fittedValues, filename):
        '''
        function writes output into yaml file, statistical analysis of the result
        '''
        values = dict(fittedValues.params.valuesdict())
        for key in values:
            values[key] = values.get(key).item()  # converdion into python native type
        data = {str(filename): values}
        outfile = open(self.directory + 'outfile.yaml', 'a')  # append to the already existing file, or create it, if ther is no file
        yaml.dump(data, outfile)
        print(yaml.dump(data))

    def prepare_txt(self):
        self.trace_b = 'TRACE: B'
        self.skiprows_txt = 21  # header rows inside the *.txt files
        self.skiprows_trace = 2  # line between traces blocks

    def readin_Data_from_file(self, filename, max_rows):
        """
        data from txt files get read in, basic calculations for complex Z, Y, epsilon and k are done
        """
        logger.debug('going to process file: ' + self.directory + filename)
        txt_file = open(self.directory + filename, 'r')
        try:
            fileDataArray = np.loadtxt(txt_file, delimiter='\t', skiprows=self.skiprows_txt, max_rows=max_rows)
        except ValueError as v:
            logger.error('Error in file ' + filename, v.arg)
        fileDataArray = np.array(fileDataArray)  # convert into numpy array
        filteredvalues = np.empty((0, fileDataArray.shape[1]))
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
        """
        epsilon = (Y - 1j * omega * constants.cf) / (1j * omega * constants.c0)
        k = -epsilon.imag * omega * constants.e0
        txt_file.close()
        return([omega, Z, Y, epsilon, k])
        """
        return([omega, Z])

    def process_data_from_file(self, dataArray, filename):
        """
        fitting the data to the cole_cole_model first(compensation of the electrode polarization)
        """
        if self.mode == 'Matlab':
            cole_cole_output = self.fit_to_cole_cole(dataArray[0], dataArray[3].real, dataArray[4])
            if self.LogLevel == 'DEBUG':
                suspension_output = self.fit_to_suspension_model(dataArray[0], dataArray[3].real, dataArray[4])
        else:
            cole_cole_output = self.fit_to_cole_cole(dataArray[0], dataArray[1])
            if self.LogLevel == 'DEBUG':
                suspension_output = self.fit_to_suspension_model(dataArray[0], dataArray[1])
        if self.LogLevel == 'DEBUG':
            self.write_suspension_output(dataArray[0], cole_cole_output.params.valuesdict(), filename)
            self.plot_cole_cole(dataArray[0], dataArray[1], cole_cole_output, filename)
            self.plot_dielectric_properties(dataArray[0], cole_cole_output, suspension_output)
        k_fit = cole_cole_output.params.valuesdict()['k']
        alpha_fit = cole_cole_output.params.valuesdict()['alpha']
        # now we need to know k and alpha only
        # fit data to cell model
        if self.model == 'SingleShell':
            single_shell_output = self.fit_to_single_shell(dataArray[0], dataArray[1], k_fit, alpha_fit)
            fit_output = single_shell_output
        else:
            double_shell_output = self.fit_to_double_shell(dataArray[0], dataArray[1], k_fit, alpha_fit)
            fit_output = double_shell_output
        if self.LogLevel == 'DEBUG':
            if self.model == 'SingleShell':
                self.plot_single_shell(dataArray[0], dataArray[1], single_shell_output, filename)
            else:
                self.plot_double_shell(dataArray[0], dataArray[1], double_shell_output, filename)
        return fit_output

    def plot_dielectric_properties(self, omega, cole_cole_output, suspension_output):
        '''
        eps is the complex valued permitivity from which we extract relative permitivity and conductivity
        compares the dielectric properties before and after the compensation
        '''
        eh = suspension_output.params.valuesdict()['eh']
        el = suspension_output.params.valuesdict()['epsi_l']
        tau = suspension_output.params.valuesdict()['tau']
        a = suspension_output.params.valuesdict()['a']
        eps_fit = self.e_sus(omega, eh, el, tau, a)
        eps_r_fit, cond_fit = self.return_diel_properties(omega, eps_fit)
        eh = cole_cole_output.params.valuesdict()['eh']
        el = cole_cole_output.params.valuesdict()['epsi_l']
        tau = cole_cole_output.params.valuesdict()['tau']
        a = cole_cole_output.params.valuesdict()['a']
        eps_fit_corrected = self.e_sus(omega, eh, el, tau, a)
        eps_r_fit_corr, cond_fit_corr = self.return_diel_properties(omega, eps_fit_corrected)
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.title('dielectric properties compared')
        plt.plot(omega, eps_r_fit, label="eps_r_fit")
        plt.plot(omega, eps_r_fit_corr, label="eps_r_fit_corr")
        plt.legend()

        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.title('dielectric properties compared')
        plt.plot(omega, cond_fit, label="cond_fit")
        plt.plot(omega, cond_fit_corr, label="cond_fit_corr")
        plt.legend()

    def write_suspension_output(self, omega, values, filename):
        '''
        writes the output of the suspension_fit into a txt file
        '''
        eh = values['eh']
        el = values['epsi_l']
        tau = values['tau']
        a = values['a']
        es = eh + (el - eh) / (1. + (1j * omega * tau) ** a)
        es = np.column_stack((es.real, es.imag))
        out = np.column_stack((omega, es))
        np.savetxt('ES_' + filename + '.txt', out)

    def select_and_solve(self, solvername, residual, params, args):
        '''
        selects the needed solver and minimizes to the given residual
        '''
        if(solvername == 'basinhopping'):
            return minimize(residual, params, args=args, method='basinhopping')
        if(solvername == 'levenberg-marquardt'):
            return minimize(residual, params, args=args, method='leastsq', ftol=1e-12, xtol=1e-12, maxfev=10000)
        if(solvername == 'least-squares'):
            return minimize(residual, params, args=args, method='least_squares')  # , ftol=1e-12, xtol=1e-12, max_nfev=4000)
        if(solvername == 'nelder-mead'):
            return minimize(residual, params, args=args, method='nelder', options={'adaptive': True})
        if(solvername == 'differential_evolution'):
            return minimize(residual, params, args=args, method='differential_evolution')

    def compare_to_data(self, omega, Z, Z_fit, filename):
        '''
        plots the relative difference of the fitted function to the data
        '''
        plt.figure()
        plt.xscale('log')
        plt.title(str(filename) + "relative difference to data")
        plt.ylabel('rel. difference [%] to data')
        plt.plot(omega, 100. * np.abs((Z.real - Z_fit.real) / Z.real), 'g', label='rel .difference real part')
        plt.plot(omega, 100. * np.abs((Z.imag - Z_fit.imag) / Z.imag), 'r', label='rel. difference imag part')
        plt.legend()

    def set_parameters_from_yaml(self, params, modelName):
        if(modelName == 'single_shell'):
            single_shell_input = open('single_shell_input.yaml', 'r')
            bufdict = yaml.safe_load(single_shell_input)
        if(modelName == 'double_shell'):
            double_shell_input = open('double_shell_input.yaml', 'r')
            bufdict = yaml.safe_load(double_shell_input)
        if(modelName == 'cole_cole'):
            cole_cole_input = open('cole_cole_input.yaml', 'r')
            bufdict = yaml.safe_load(cole_cole_input)
        if(modelName == 'suspension'):
            suspension_input = open('suspension_input.yaml', 'r')
            bufdict = yaml.safe_load(suspension_input)
        for key in bufdict:
            params.add(key, value=float(bufdict[key]['value']), min=float(bufdict[key]['min']), max=float(bufdict[key]['max']), vary=bool(bufdict[key]['vary']))
        return params

    def Z_CPE(self, omega, k, alpha):
        return (1. / k) * (1j * omega) ** (-alpha)

    def Z_sus(self, omega, es, kdc):  # only valid for cole_cole_fit and suspension_fit
        return 1. / (1j * es * omega * constants.c0 + (kdc * constants.c0) / constants.e0 + 1j * omega * constants.cf)

    def e_sus(self, omega, eh, el, tau, a):  # this is only valid for the cole_cole_fit and suspension_fit
        return eh + (el - eh) / (1. + (1j * omega * tau) ** (a))

    def return_diel_properties(self, omega, epsc):
        eps_r = epsc.real
        conductivity = -epsc.imag * constants.e0 * omega
        return eps_r, conductivity

    #######################################################################
    # suspension model section
    def fit_to_suspension_model(self, omega, input1, k=None):
        '''
        input 1 is either  Z in case of mode =='Matlab' or the real part of epsilon
        '''
        logger.debug('fit data to pure suspension model')
        params = Parameters()
        params = self.set_parameters_from_yaml(params, 'suspension')
        result = None
        if k is None:
            result = self.select_and_solve(self.solvername, self.suspension_residual, params, (omega, input1))
            # ValueError: differential_evolution requires finite bound for all varying parameters
            # for differential_evolution, but all bounds are set => no. Some bounds are set to Infinity. See the parameters file
        else:
            result = self.select_and_solve(self.solvername, self.suspension_residual, params, (omega, input1, k))
        logger.info(lmfit.fit_report(result))
        logger.info(result.params.pretty_print())
        return result

    def suspension_residual(self, params, omega, data, data_i=None):
        el = params['epsi_l'].value
        tau = params['tau'].value
        a = params['a'].value
        kdc = params['conductivity'].value
        eh = params['eh'].value
        Z_fit = self.suspension_model(omega, el, tau, a, kdc, eh)
        if data_i is None:
            residual = data - Z_fit
            return residual.view(np.float)
        else:
            Y_fit = 1. / Z_fit
            efit = (Y_fit / (1j * omega * constants.c0))
            efit_r = efit.real
            efit_i = -efit.imag
            kfit = efit_i * omega * constants.e0
            residual = np.concatenate((1. - (np.log10(data) / np.log10(efit_r)), 1. - (data_i / kfit)))
            return residual

    def suspension_model(self, omega, el, tau, a, kdc, eh):
        es = self.e_sus(omega, eh, el, tau, a)
        Zs_fit = self.Z_sus(omega, es, kdc)
        Z_fit = Zs_fit
        return Z_fit

    ################################################################
    #  cole_cole_section
    def fit_to_cole_cole(self, omega, input1, k=None):
        # find more details in formula 6 and 7 of https://ieeexplore.ieee.org/document/6191683
        '''
        input 1 is either  Z in case of mode =='Matlab' or the real part of epsilon
        '''
        logger.debug('fit data to cole-cole model to account for electrode polarisation')
        params = Parameters()
        params = self.set_parameters_from_yaml(params, 'cole_cole')
        result = None
        for i in range(2):  # we have two iterations
            if i == 1:
                # fix these two parameters, conductivity and eh
                params['conductivity'].set(vary=False, value=result.params['conductivity'].value)
                params['eh'].set(vary=False, value=result.params['eh'].value)
            if k is None:
                result = self.select_and_solve(self.solvername, self.cole_cole_residual, params, args=(omega, input1))
            else:
                result = self.select_and_solve(self.solvername, self.cole_cole_residual, params, args=(omega, input1, k))
            # logger.info(lmfit.fit_report(result.params))
            logger.info(lmfit.fit_report(result))
            logger.debug(result.params.pretty_print())
        global emed
        emed = result.params['eh'].value  # emed = eh from first fit
        return result

    def cole_cole_residual(self, params, omega, data, data_i=None):
        """
        We have two ways to compute the residual. One as in the Matlab script (data_i is not None) and just a plain fit.
        In the matlab case, the input parameters have to be epsilon.real and k, which are determined in readin_Data_from_file.
        """
        k = params['k'].value
        el = params['epsi_l'].value
        tau = params['tau'].value
        a = params['a'].value
        alpha = params['alpha'].value
        kdc = params['conductivity'].value
        eh = params['eh'].value
        Z_fit = self.cole_cole_model(omega, k, el, tau, a, alpha, kdc, eh)
        if data_i is None:
            residual = data - Z_fit
            return residual.view(np.float)
        else:
            Y_fit = 1. / Z_fit
            efit = (Y_fit / (1j * omega * constants.c0))
            efit_r = efit.real
            efit_i = -efit.imag
            kfit = efit_i * omega * constants.e0
            residual = np.concatenate((1. - (np.log10(np.abs(data)) / np.log10(efit_r)), 1. - (data_i / kfit)))
            # data was negative for one of the xlxs files, so the abs is drawn
            return residual

    def cole_cole_model(self, omega, k, el, tau, a, alpha, kdc, eh):

        """
        function holding the cole_cole_model equations, returning the calculated impedance
        """
        Zep_fit = self.Z_CPE(omega, k, alpha)
        es = self.e_sus(omega, eh, el, tau, a)

        Zs_fit = self.Z_sus(omega, es, kdc)
        Z_fit = Zep_fit + Zs_fit

        return Z_fit

    def plot_cole_cole(self, omega, Z, result, filename):
        popt = np.fromiter(result.params.valuesdict().values(), dtype=np.float)
        Z_fit = self.cole_cole_model(omega, *popt)
        if self.LogLevel == "DEBUG" and self.mode == 'Matlab':
            csvfile = open("1610_matlab_fit_cole_cole.csv", newline="\n")
            matlabfitData = list(csv.reader(csvfile, delimiter=','))  # create a list, otherwise, its just  the iterator over the file
            matlabfitData = np.array(matlabfitData, dtype=np.float)  # convert into numpy array

        # plot real  Impedance part
        plt.xscale('log')
        plt.title(str(filename) + " Z_real_part")
        plt.plot(omega, Z_fit.real, '+', label='fitted by Python')
        plt.plot(omega, Z.real, 'r', label='data')
        plt.legend()
        # plot imaginaray Impedance part
        plt.figure()
        plt.title(str(filename) + " Z_imaginary_part")
        plt.xscale('log')
        plt.plot(omega, Z_fit.imag, '+', label='fitted by Python')
        plt.plot(omega, Z.imag, 'r', label='data')
        plt.legend()
        # plot real vs  imaginary Partr
        plt.figure()
        plt.title(str(filename) + " real vs imag")
        plt.plot(Z_fit.real, Z_fit.imag, '+', label="Z_fit")
        plt.plot(Z.real, Z.imag, 'o', label="Z")
        plt.legend()
        self.compare_to_data(omega, Z, Z_fit, filename)
        plt.show()

    #################################################################
    # single_shell_section
    def fit_to_single_shell(self, omega, input1, k_fit, alpha_fit, k=None):
        '''
        input and k depend on the mode being used, it is either Z if k is none or the real Part of epsilon
        Mode=='Matlab' uses k and e.real
        '''
        params = Parameters()
        params = self.set_parameters_from_yaml(params, 'single_shell')
        params.add('k', vary=False, value=k_fit)
        params.add('alpha', vary=False, value=alpha_fit)
        result = None
        if k is None:
            result = self.select_and_solve(self.solvername, self.single_shell_residual, params, args=(omega, input1))
        else:
            result = self.select_and_solve(self.solvername, self.single_shell_residual, self.params, args=(omega, input1, k))
        logger.info(lmfit.fit_report(result))
        logger.info(result.message)
        logger.debug((result.params.pretty_print()))
        return result

    def single_shell_model(self, omega, k, alpha, em, km, kcp, kmed):
        '''
        formulas for the single shell model are described here
             -returns: Calculated impedance corrected by the constant-phase-element
        '''
        epsi_cp = constants.ecp - 1j * kcp / (constants.e0 * omega)
        epsi_m = em - 1j * km / (constants.e0 * omega)
        epsi_med = constants.emed - 1j * kmed / (constants.e0 * omega)
        # model
        E1 = epsi_cp / epsi_m
        epsi_cell = epsi_m * (2. * (1. - constants.v1) + (1. + 2. * constants.v1) * E1) / ((2. + constants.v1) + (1. - constants.v1) * E1)

        # electrode polarization and calculation of Z
        E0 = epsi_cell / epsi_med
        esus = epsi_med * (2. * (1. - constants.p) + (1. + 2. * constants.p) * E0) / ((2. + constants.p) + (1. - constants.p) * E0)
        Ys = 1j * esus * omega * constants.c0 + 1j * omega * constants.cf                 # cell suspension admittance spectrum
        Zs = 1 / Ys
        Zep = self.Z_CPE(omega, k, alpha)               # including EP
        Z = Zs + Zep
        return Z

    def single_shell_residual(self, params, omega, data, data_i=None):
        '''
        calculates the residual for the single-shell model, using the single_shell_model.
        if the
        Mode=='Matlab' uses k and e.real as data and data_i
        :param data:
            if data_i is not give, the array in data will be the Impedance calculated from the input file
            if data_i is given, it will be the real part of the permitivity calculated from the input file
        :param data_i:
            if this is given, it will be the conductivity calculated from the input file
        :param params:
            parameters used fot the calculation of the impedance
        '''
        k = params['k'].value
        alpha = params['alpha'].value
        em = params['em'].value
        km = params['km'].value
        kcp = params['kcp'].value
        kmed = params['kmed'].value
        Z_fit = self.single_shell_model(omega, k, alpha, em, km, kcp, kmed)
        if data_i is None:
            residual = data - Z_fit
            return residual.view(np.float)
        else:
            Y_fit = 1. / Z_fit
            efit = (Y_fit / (1j * omega * constants.c0))
            efit_r = efit.real
            efit_i = -efit.imag
            kfit = efit_i * omega * constants.e0
            residual = np.concatenate((1. - (np.log10(np.abs(data)) / np.log10(efit_r)), 1. - (data_i / kfit)))
            return residual

    def plot_single_shell(self, omega, Z, result, filename):
        '''
        plot the real part, imaginary part vs frequency and real vs. imaginary part
        '''
        # calculate fitted Z function
        popt = np.fromiter(result.params.valuesdict().values(), dtype=np.float)
        Z_fit = self.single_shell_model(omega, *popt)

        # plot real  Impedance part
        plt.figure()
        plt.xscale('log')
        plt.title(str(filename) + " Z_real_part_single_shell")
        plt.plot(omega, Z_fit.real, '+', label='fitted by Python')
        plt.plot(omega, Z.real, 'r', label='data')
        plt.legend()
        # plot imaginaray Impedance part
        plt.figure()
        plt.title(str(filename) + " Z_imaginary_part_single_shell")
        plt.xscale('log')
        plt.plot(omega, Z_fit.imag, '+', label='fitted by Python')
        plt.plot(omega, Z.imag, 'r', label='data')
        plt.legend()
        # plot real vs  imaginary Partr
        plt.figure()
        plt.title(str(filename) + " real vs imag single_shell")
        plt.plot(Z_fit.real, Z_fit.imag, '+', label="Z_fit")
        plt.plot(Z.real, Z.imag, 'o', label="Z")
        plt.legend()

        self.compare_to_data(omega, Z, Z_fit, filename)
        plt.show()

    ################################################################
    # double_shell_section
    def fit_to_double_shell(self, omega, input1, k_fit, alpha_fit, k=None):
        '''
        input 1 can either be Z, if  k is None, or  the real Part of epsilon, k_fit and alpha_fit
        are the determined values in the cole-cole-fit
        '''
        logger.debug('fit data to double shell model')
        params = Parameters()
        params = self.set_parameters_from_yaml(params, 'double_shell')
        params.add('k', vary=False, value=k_fit)
        params.add('alpha', vary=False, value=alpha_fit)

        result = None
        for i in range(4):
            if i == 1:
                params['kmed'].set(vary=False, value=result.params.valuesdict()['kmed'])
            if i == 2:
                params['km'].set(vary=False, value=result.params.valuesdict()['km'])
                params['em'].set(vary=False, value=result.params.valuesdict()['em'])
            if i == 3:
                params['kcp'].set(vary=False, value=result.params.valuesdict()['kcp'])
            if k is None:
                result = self.select_and_solve(self.solvername, self.double_shell_residual, params, args=(omega, input1))
            else:
                result = self.select_and_solve(self.solvername, self.double_shell_residual, params, args=(omega, input1, k))
            logger.info(lmfit.fit_report(result))
            logger.info(result.message)
            logger.debug(result.params.pretty_print())
        return result

    def double_shell_residual(self, params, omega, data, data_i=None):
        '''
        data is Z if data_i is None, if data_i is k, the real part of epsilon is data
        '''
        k = params['k'].value
        alpha = params['alpha'].value
        km = params['km'].value
        em = params['em'].value
        kcp = params['kcp'].value
        ene = params['ene'].value
        kne = params['kne'].value
        knp = params['knp'].value
        kmed = params['kmed'].value

        Z_fit = self.double_shell_model(omega, k, alpha, km, em, kcp, ene, kne, knp, kmed)
        # define the objective function
        # optimize for impedance
        if data_i is None:
            residual = data - Z_fit
            return residual.view(np.float)
        # optimize for k and e.real
        else:
            Y_fit = 1. / Z_fit
            efit = (Y_fit / (1j * omega * constants.c0))
            efit_r = efit.real
            efit_i = -efit.imag
            kfit = efit_i * omega * constants.e0
            residual = np.concatenate((1. - (np.log10(np.abs(data)) / np.log10(efit_r)), 1. - (data_i / kfit)))
            return residual

    def double_shell_model(self, omega, k, alpha, km, em, kcp, ene, kne, knp, kmed):
        """
        atm, this is a dummy for the Double-Shell-Model equations,
        containing everything from double_shell_sopt_pso.py to fit the variable definitions in this file
        """
        epsi_m = em + km / (1j * omega * constants.e0)
        epsi_cp = constants.ecp + kcp / (1j * omega * constants.e0)
        epsi_ne = ene + kne / (1j * omega * constants.e0)
        epsi_np = constants.enp + knp / (1j * omega * constants.e0)
        epsi_med = emed + kmed / (1j * omega * constants.e0)

        E3 = epsi_np / epsi_ne
        E2 = ((epsi_ne / epsi_cp) * (2. * (1. - constants.v3) + (1. + 2. * constants.v3) * E3) /
              ((2. + constants.v3) + (1. - constants.v3) * E3))  # Eq. 13
        E1 = ((epsi_cp / epsi_m) * (2. * (1. - constants.v2) + (1. + 2. * constants.v2) * E2) /
              ((2. + constants.v2) + (1. - constants.v2) * E2))  # Eq. 14

        epsi_cell = (epsi_m * (2. * (1. - constants.v1) + (1. + 2. * constants.v1) * E1) /
                     ((2. + constants.v1) + (1. - constants.v1) * E1))  # Eq. 11
        E0 = epsi_cell / epsi_med
        esus = epsi_med * (2. * (1. - constants.p) + (1. + 2. * constants.p) * E0) / ((2. + constants.p) + (1. - constants.p) * E0)
        Ys = 1j * esus * omega * constants.c0 + 1j * omega * constants.cf                 # cell suspension admittance spectrum
        Zs = 1 / Ys
        Zep = self.Z_CPE(omega, k, alpha)               # including EP
        Z = Zs + Zep
        return Z

    def plot_double_shell(self, omega, Z, result, filename):
        '''
        plot the real and imaginary part of the impedance vs. the frequency and
        real vs. imaginary part
        '''
        popt = np.fromiter(result.params.valuesdict().values(), dtype=np.float)
        Z_fit = self.double_shell_model(omega, *popt)

        # plot real  Impedance part
        plt.figure()
        plt.xscale('log')
        plt.title(str(filename) + " Z_real_part double_shell")
        plt.plot(omega, Z_fit.real, '+', label='fitted by Python')
        plt.plot(omega, Z.real, 'r', label='data')
        plt.legend()
        # plot imaginaray Impedance part
        plt.figure()
        plt.title(str(filename) + " Z_imaginary_part double shell")
        plt.xscale('log')
        plt.plot(omega, Z_fit.imag, '+', label='fitted by Python')
        plt.plot(omega, Z.imag, 'r', label='data')
        plt.legend()
        # plot real vs  imaginary Partr
        plt.figure()
        plt.title(str(filename) + " real vs imag double shell")
        plt.plot(Z_fit.real, Z_fit.imag, '+', label="Z_fit")
        plt.plot(Z.real, Z.imag, 'o', label="Z")
        plt.legend()
        self.compare_to_data(omega, Z, Z_fit, filename)
        plt.show()
