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
from copy import deepcopy
from .single_shell import plot_single_shell, single_shell_residual
from .double_shell import plot_double_shell, double_shell_residual
from .cole_cole import plot_cole_cole, cole_cole_residual, suspension_residual
from .cole_cole_R import plot_cole_cole_R, cole_cole_R_residual
from .rc import plot_rc, rc_residual
from .RC import plot_RC, RC_residual
from .utils import set_parameters, plot_dielectric_properties, get_labels

# create logger
logger = logging.getLogger('impedancefitter-logger')
"""
We use an own logger for the impedancefitter module.
"""


class Fitter(object):
    """
    The main fitting object class.
    The chosen fitting routine is applied to all files in the chosen directory.

    Parameters
    ----------

    directory: {string, optional, path to data directory}
        provide directory if you run code from directory different to data directory
    parameters: {dict, optional, needed parameters}
        provide parameters if you do not want to use a yaml file (for instance in parallel UQ runs).
    write_output: {bool, optional}
        decide if you want to dump output to file
    compare_CPE: {bool, optional}
        decide if you want to compare to case without CPE

    Kwargs
    ------

    model: {string, 'SingleShell' OR 'DoubleShell'}
        currently, you can choose between SingleShell and DoubleShell.
    solvername : {string, name of solver}
        choose among global solvers from lmfit: basinhopping, differential_evolution, ...; local solvers: levenberg-marquardt, nelder-mead, least-squares. See also lmfit documentation.
    inputformat: {string, 'TXT' OR 'XLSX'}
        currently only TXT and Excel files with a certain formatting (impedance in two columns, first is real part, second imaginary part, is accepted).
    LogLevel: {string, optional, 'DEBUG' OR 'INFO' OR 'WARNING'}
        choose level for logger. Case DEBUG: the script will output plots after each fit, case INFO: the script will output results from each fit to the console.
    excludeEnding: {string, optional}
        for ending that should be ignored (if there are files with the same ending as the chosen inputformat)
    minimumFrequency: {float, optional}
        if you want to use another frequency than the minimum frequency in the dataset.
    maximumFrequency: {float, optional}
        if you want to use another frequency than the maximum frequency in the dataset.
    solver_kwargs: {dict, optional}
        Contains infos for the solver. find possible kwargs in the lmfit documentation.
    data_sets: {int, optional}
        Use only a certain number of data sets instead of all in directory.
    current_threshold: {float, optional}
        use only for data from E4980AL LCR meter to check current

    Attributes
    ----------

    omega: list
        Contains frequencies.
    Z: list
        Contains corresponding impedances.
    protocol: None or string
        Choose 'Iterative' for repeated fits with changing parameter sets, customized approach. If not specified, there is always just one fit for each data set.
    """

    def __init__(self, directory=None, parameters=None, write_output=True, compare_CPE=False, fileList=None, **kwargs):
        self.model = kwargs['model']
        self.solvername = kwargs['solvername']
        self.inputformat = kwargs['inputformat']
        self.write_output = write_output
        self.compare_CPE = compare_CPE
        self.fileList = fileList
        try:
            self.LogLevel = kwargs['LogLevel']  # log level: choose info for less verbose output
        except KeyError:
            self.LogLevel = 'INFO'
            pass
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
            pass
        if directory is None:
            directory = os.getcwd()
        if 'current_threshold' in kwargs:
            self.current_threshold = kwargs['current_threshold']
        else:
            self.current_threshold = None

        self.directory = directory + '/'
        logger.setLevel(self.LogLevel)

        if not len(logger.handlers):
            # create console handler and set level to debug
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            logger.addHandler(ch)

        if parameters is not None:
            assert(isinstance(parameters, dict)), "You need to provide an input dictionary!"
        self.parameters = deepcopy(parameters)

    def initialize_parameters(self):
        if self.model == 'ColeCole':
            self.cole_cole_parameters = Parameters()
            self.cole_cole_parameters = set_parameters(self.cole_cole_parameters, 'cole_cole', self.parameters, ep=self.electrode_polarization, ind=self.inductivity, loss=self.high_loss)
            if self.LogLevel == 'DEBUG':
                self.suspension_parameters = Parameters()
                self.suspension_parameters = set_parameters(self.suspension_parameters, 'suspension', self.parameters)
        elif self.model == 'ColeColeR':
            self.cole_cole_parameters = Parameters()
            self.cole_cole_parameters = set_parameters(self.cole_cole_parameters, 'cole_cole_r', self.parameters, ep=self.electrode_polarization, ind=self.inductivity, loss=self.high_loss)
        elif self.model == 'Randles':
            self.randles_parameters = Parameters()
            self.randles_parameters = set_parameters(self.randles_parameters, 'randles', self.parameters, ep=False, ind=self.inductivity, loss=self.high_loss)
        elif self.model == 'Randles_CPE':
            self.randles_parameters = Parameters()
            self.randles_parameters = set_parameters(self.randles_parameters, 'randles_cpe', self.parameters, ep=False, ind=self.inductivity, loss=self.high_loss)
        elif self.model == 'RC':
            self.rc_parameters = Parameters()
            self.rc_parameters = set_parameters(self.rc_parameters, 'rc', self.parameters, ep=self.electrode_polarization, ind=self.inductivity, loss=self.high_loss)
        elif self.model == 'RC_full':
            self.RC_parameters = Parameters()
            self.RC_parameters = set_parameters(self.RC_parameters, 'RC', self.parameters, ep=self.electrode_polarization, ind=self.inductivity, loss=self.high_loss)
        elif self.model == 'SingleShell':
            if self.electrode_polarization_fit is True:
                self.cole_cole_parameters = Parameters()
                self.cole_cole_parameters = set_parameters(self.cole_cole_parameters, 'cole_cole', self.parameters, ep=self.electrode_polarization_fit, ind=self.inductivity, loss=self.high_loss)
                if self.LogLevel == 'DEBUG':
                    self.suspension_parameters = Parameters()
                    self.suspension_parameters = set_parameters(self.suspension_parameters, 'suspension', self.parameters)
            self.single_shell_parameters = Parameters()
            self.single_shell_parameters = set_parameters(self.single_shell_parameters, 'single_shell', self.parameters, ep=self.electrode_polarization)
        elif self.model == 'DoubleShell':
            if self.electrode_polarization_fit is True:
                self.cole_cole_parameters = Parameters()
                self.cole_cole_parameters = set_parameters(self.cole_cole_parameters, 'cole_cole', self.parameters, ep=self.electrode_polarization_fit, ind=self.inductivity, loss=self.high_loss)
                if self.LogLevel == 'DEBUG':
                    self.suspension_parameters = Parameters()
                    self.suspension_parameters = set_parameters(self.suspension_parameters, 'suspension', self.parameters)
            self.double_shell_parameters = Parameters()
            self.double_shell_parameters = set_parameters(self.double_shell_parameters, 'double_shell', self.parameters, ep=self.electrode_polarization)

    def main(self, protocol=None, electrode_polarization_fit=False, electrode_polarization=False, inductivity=False, high_loss=False):
        """
        Main function that iterates through all data sets provided.

        Parameters
        ----------

        protocol: None or string
            Choose 'Iterative' for repeated fits with changing parameter sets, customized approach. If not specified, there is always just one fit for each data set.

        electrode_polarization_fit: True or False
            Switch on whether to account for electrode polarization or not. Currently, only a CPE correction is possible. If True, a Cole-Cole model is used for fitting.

        electrode_polarization: True or False
            Switch on whether to account for electrode polarization or not. Currently, only a CPE correction is possible.
        """
        max_rows_tag = False
        self.electrode_polarization = electrode_polarization
        self.electrode_polarization_fit = electrode_polarization_fit
        self.inductivity = inductivity
        self.high_loss = high_loss
        self.initialize_parameters()
        self.protocol = protocol
        if self.write_output is True:
            open('outfile.yaml', 'w')  # create output file
        if self.fileList is None:
            self.fileList = os.listdir(self.directory)
        for filename in self.fileList:
            filename = os.fsdecode(filename)
            if filename.endswith(self.excludeEnding):
                logger.info("Skipped file {} due to excluded ending.".format(filename))
                continue
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
            elif self.inputformat == 'CSV' and filename.endswith(".csv"):
                self.readin_Data_from_csv(filename)
                iters = len(self.zarray)
                if self.data_sets is not None:
                    iters = self.data_sets
                for i in range(iters):
                    self.Z = self.zarray[i]
                    self.fittedValues = self.process_data_from_file(filename)
                    self.process_fitting_results(filename + ' Row' + str(i))
            elif self.inputformat == 'CSV_E4980AL' and filename.endswith(".csv"):
                self.readin_Data_from_csv_E4980AL(filename)
                iters = len(self.zarray)
                if self.data_sets is not None:
                    iters = self.data_sets
                for i in range(iters):
                    self.Z = self.zarray[i]
                    self.fittedValues = self.process_data_from_file(filename)
                    self.process_fitting_results(filename)

    def process_fitting_results(self, filename):
        '''
        function writes output into yaml file to prepare statistical analysis of the result
        '''
        values = dict(self.fittedValues.params.valuesdict())
        for key in values:
            values[key] = float(values[key])  # conversion into python native type
        self.data = {str(filename): values}
        if self.write_output is True:
            outfile = open('outfile.yaml', 'a')  # append to the already existing file, or create it, if ther is no file
            yaml.dump(self.data, outfile)

    def prepare_txt(self):
        """
        Function for txt files that are currently supported. The section for "TRACE: A" is used, also the number of skiprows_txt needs to be aligned, if there is a deviating TXT-format.
        """
        self.trace_b = 'TRACE: B'
        self.skiprows_txt = 21  # header rows inside the *.txt files
        self.skiprows_trace = 2  # line between traces blocks

    def readin_Data_from_file(self, filename, max_rows):
        """
        Data from txt files get reads in, returns array with omega and complex-valued impedance Z
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
        fit the data to the cole_cole_model first (compensation of the electrode polarization) and then to the defined model.
        """
        if self.model == 'RC':
            fit_output = self.fit_to_rc(self.omega, self.Z)
        elif self.model == 'RC_full':
            fit_output = self.fit_to_RC(self.omega, self.Z)
        elif self.model == 'ColeCole':
            fit_output = self.fit_to_cole_cole(self.omega, self.Z)
        elif self.model == 'ColeColeR':
            fit_output = self.fit_to_cole_cole_r(self.omega, self.Z)
        # now we need to know k and alpha only
        # fit data to cell model
        if self.model == 'SingleShell' or self.model == 'DoubleShell':
            if self.electrode_polarization_fit is True:
                self.cole_cole_output = self.fit_to_cole_cole(self.omega, self.Z)
                k_fit = self.cole_cole_output.params.valuesdict()['k']
                alpha_fit = self.cole_cole_output.params.valuesdict()['alpha']
                if self.LogLevel == 'DEBUG':
                    plot_cole_cole(self.omega, self.Z, self.cole_cole_output, filename)
                if self.compare_CPE is True:
                    suspension_output = self.fit_to_suspension_model(self.omega, self.Z)
                    plot_dielectric_properties(self.omega, self.cole_cole_output, suspension_output)
            else:
                k_fit = None
                alpha_fit = None
        if self.model == 'SingleShell':
            fit_output = self.fit_to_single_shell(self.omega, self.Z, k_fit, alpha_fit)
        elif self.model == 'DoubleShell':
            fit_output = self.fit_to_double_shell(self.omega, self.Z, k_fit, alpha_fit)
        if self.LogLevel == 'DEBUG':
            if self.model == 'SingleShell':
                plot_single_shell(self.omega, self.Z, fit_output, filename)
            elif self.model == 'DoubleShell':
                plot_double_shell(self.omega, self.Z, fit_output, filename)
            elif self.model == 'RC':
                plot_rc(self.omega, self.Z, fit_output, filename)
            elif self.model == 'RC_full':
                plot_RC(self.omega, self.Z, fit_output, filename)
            elif self.model == 'ColeCole':
                plot_cole_cole(self.omega, self.Z, fit_output, filename)
            elif self.model == 'ColeColeR':
                plot_cole_cole_R(self.omega, self.Z, fit_output, filename)
        return fit_output

    def select_and_solve(self, solvername, residual, params, args):
        '''
        selects the needed solver and minimizes the given residual
        '''
        self.solver_kwargs['method'] = solvername
        self.solver_kwargs['args'] = args
        return minimize(residual, params, **self.solver_kwargs)

    #######################################################################
    #  rc_section
    def fit_to_rc(self, omega, Z):
        '''
        Fit data to RC model.
        '''
        logger.debug('####################')
        logger.debug('fit data to RC model')
        logger.debug('####################')
        params = deepcopy(self.rc_parameters)
        result = None
        iters = 1
        for i in range(iters):
            logger.info("###########\nFitting round {}\n###########".format(i + 1))
            result = self.select_and_solve(self.solvername, rc_residual, params, args=(omega, Z))
            logger.info(lmfit.fit_report(result))
            if self.solvername != "ampgo":
                logger.info(result.message)
            else:
                logger.info(result.ampgo_msg)
            logger.debug(result.params.pretty_print())
        return result

    def fit_to_RC(self, omega, Z):
        '''
        Fit data to full RC model without known c0.
        '''
        logger.debug('#########################')
        logger.debug('fit data to full RC model')
        logger.debug('#########################')
        params = deepcopy(self.RC_parameters)
        result = None
        iters = 1
        for i in range(iters):
            logger.info("###########\nFitting round {}\n###########".format(i + 1))
            result = self.select_and_solve(self.solvername, RC_residual, params, args=(omega, Z))
            logger.info(lmfit.fit_report(result))
            if self.solvername != "ampgo":
                logger.info(result.message)
            else:
                logger.info(result.ampgo_msg)
            logger.debug(result.params.pretty_print())
        return result

    #######################################################################
    # suspension model section
    def fit_to_suspension_model(self, omega, Z):
        '''
        fit data to pure suspension model to compare to cole-cole based correction.
        See also: :func:`impedancefitter.cole_cole.suspension_model`

        '''
        logger.debug('#################################')
        logger.debug('fit data to pure suspension model')
        logger.debug('#################################')
        params = deepcopy(self.suspension_parameters)
        result = None
        result = self.select_and_solve(self.solvername, suspension_residual, params, (omega, Z))
        logger.info(lmfit.fit_report(result))
        logger.info(result.params.pretty_print())
        return result

    ################################################################
    #  cole_cole_section
    def fit_to_cole_cole(self, omega, Z):
        '''
        Fit data to cole-cole model.

        If the :attr:`protocol` is specified to be 'Iterative', the data is fitted to the Cole-Cole-Model twice. Then, in the second fit, the parameters 'conductivity' and 'eh' are being fixed.

        The initial parameters and boundaries for the variables are read from the .yaml file in the directory of the python script.
        see also: :func:`impedancefitter.cole_cole.cole_cole_model`
        '''
        logger.debug('#################################################################')
        logger.debug('fit data to cole-cole model to account for electrode polarisation')
        logger.debug('#################################################################')
        params = deepcopy(self.cole_cole_parameters)
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
            result = self.select_and_solve(self.solvername, cole_cole_residual, params, args=(omega, Z))
            logger.info(lmfit.fit_report(result))
            if self.solvername != "ampgo":
                logger.info(result.message)
            else:
                logger.info(result.ampgo_msg)
            logger.debug(result.params.pretty_print())
        return result

    def fit_to_cole_cole_r(self, omega, Z):
        '''
        Fit data to cole-cole model.

        The initial parameters and boundaries for the variables are read from the .yaml file in the directory of the python script.
        see also: :func:`impedancefitter.cole_cole.cole_cole_model`
        '''
        logger.debug('###################################################')
        logger.debug('fit data to cole-cole model in resistor formulation')
        logger.debug('###################################################')
        params = deepcopy(self.cole_cole_parameters)
        result = None
        iters = 1
        for i in range(iters):
            logger.info("###########\nFitting round {}\n###########".format(i + 1))
            result = self.select_and_solve(self.solvername, cole_cole_R_residual, params, args=(omega, Z))
            logger.info(lmfit.fit_report(result))
            if self.solvername != "ampgo":
                logger.info(result.message)
            else:
                logger.info(result.ampgo_msg)
            logger.debug(result.params.pretty_print())
        return result

    #################################################################
    # single_shell_section
    def fit_to_single_shell(self, omega, Z, k_fit, alpha_fit):
        '''
        if :attr:`protocol` is `Iterative`, the conductivity of the medium is determined in the first run and then fixed.
        See also: :func:`impedancefitter.single_shell.single_shell_model`
        '''
        params = deepcopy(self.single_shell_parameters)
        if self.electrode_polarization_fit is True:
            params.add('k', vary=False, value=k_fit)
            params.add('alpha', vary=False, value=alpha_fit)
        assert ('emed' in params), "You need to provide emed!"
        result = None
        iters = 1
        if self.protocol == "Iterative":
            iters = 2
        for i in range(iters):  # we have two iterations
            logger.info("###########\nFitting round {}\n###########".format(i + 1))
            if i == 1:
                # fix permittivity and conductivity
                params['kmed'].set(vary=False, value=result.params.valuesdict()['kmed'])
                params['emed'].set(vary=False, value=result.params.valuesdict()['emed'])
            result = self.select_and_solve(self.solvername, single_shell_residual, params, args=(omega, Z))
            logger.info(lmfit.fit_report(result))
            if self.solvername != "ampgo":
                logger.info(result.message)
            else:
                logger.info(result.ampgo_msg)
            logger.debug((result.params.pretty_print()))
        return result

    ################################################################
    # double_shell_section
    def fit_to_double_shell(self, omega, Z, k_fit, alpha_fit):
        r"""
        k_fit and alpha_fit are the determined values in the cole-cole-fit

        If :attr:`protocol` is equal to `Iterative`, the following procedure is applied:

        #. 1st Fit: the parameters :math:`k` and :math:`\alpha` are fixed(coming from cole-cole-fit), and the data is fitted against the double-shell-model. To be determined in this fit: :math:`\sigma_\mathrm{sup} / k_\mathrm{med}`.

        #. 2nd Fit: the parameters :math:`k`, :math:`\alpha` and :math:`\sigma_\mathrm{sup}` are fixed and the data is fitted again. To be determined in this fit: :math:`\sigma_\mathrm{m}, \varepsilon_\mathrm{m}, \sigma_\mathrm{cp}`.

        #. 3rd Fit: the parameters :math:`k`, :math:`\alpha`, :math:`\sigma_\mathrm{sup}`, :math:`\sigma_\mathrm{m}` and :math:`\varepsilon_\mathrm{m}` are fixed and the data is fitted again. To be determined in this fit: :math:`\sigma_\mathrm{cp}`.

        #. last Fit: the parameters :math:`k`, :math:`\alpha`, :math:`\sigma_\mathrm{sup}, \sigma_\mathrm{ne}, \sigma_\mathrm{np}, \sigma_\mathrm{m}, \varepsilon_\mathrm{m}` are fixed. To be determined in this fit: :math:`\varepsilon_\mathrm{ne}, \sigma_\mathrm{ne}, \varepsilon_\mathrm{np}, \sigma_\mathrm{np}`.


        See also: :func:`impedancefitter.double_shell.double_shell_model`

        """
        logger.debug('##############################')
        logger.debug('fit data to double shell model')
        logger.debug('##############################')
        params = deepcopy(self.double_shell_parameters)
        if self.electrode_polarization_fit is True:
            params.add('k', vary=False, value=k_fit)
            params.add('alpha', vary=False, value=alpha_fit)
        assert ('emed' in params), "You need to provide emed!"

        result = None
        iters = 1
        if self.protocol == "Iterative":
            iters = 4
        for i in range(iters):
            logger.info("###########\nFitting round {}\n###########".format(i + 1))
            if i == 1:
                params['kmed'].set(vary=False, value=result.params.valuesdict()['kmed'])
                params['emed'].set(vary=False, value=result.params.valuesdict()['emed'])
            if i == 2:
                params['km'].set(vary=False, value=result.params.valuesdict()['km'])
                params['em'].set(vary=False, value=result.params.valuesdict()['em'])
            if i == 3:
                params['kcp'].set(vary=False, value=result.params.valuesdict()['kcp'])
            result = self.select_and_solve(self.solvername, double_shell_residual, params, args=(omega, Z))
            logger.info(lmfit.fit_report(result))
            if self.solvername != "ampgo":
                logger.info(result.message)
            else:
                logger.info(result.ampgo_msg)
            logger.debug(result.params.pretty_print())
        return result

    def emcee_plot(self, res):
        import corner
        truths = [res.params[p].value for p in res.var_names]
        ifLabels = get_labels()
        labels = [ifLabels[p] for p in res.var_names]
        plot = corner.corner(res.flatchain, labels=labels,
                             truths=truths)
        return plot

    def emcee_main(self, electrode_polarization=True, lnsigma=None, inductivity=False, high_loss=False):
        """
        Main function that iterates through all data sets provided.

        Parameters
        ----------

        electrode_polarization: True or False
            Switch on whether to account for electrode polarization or not. Currently, only a CPE correction is possible.
        lnsigma: dict
            Add lnsigma parameter with custom initial value and bounds to method.
        """
        max_rows_tag = False
        self.electrode_polarization = electrode_polarization
        self.inductivity = inductivity
        self.high_loss = high_loss
        self.lnsigma = lnsigma
        if self.write_output is True:
            open('outfile.yaml', 'w')  # create output file
        if self.fileList is None:
            self.fileList = os.listdir(self.directory)
        for filename in self.fileList:
            filename = os.fsdecode(filename)
            if filename.endswith(self.excludeEnding):
                logger.info("Skipped file {} due to excluded ending.".format(filename))
                continue
            if self.inputformat == 'TXT' and filename.endswith(".TXT"):
                self.prepare_txt()
                if max_rows_tag is False:  # all files have the same number of datarows, this only has to be determined once
                    max_rows = self.get_max_rows(filename)
                    max_rows_tag = True
                self.readin_Data_from_file(filename, max_rows)
                self.fittedValues = self.fit_emcee_models(filename)
                self.process_fitting_results(filename)
                if self.LogLevel == 'DEBUG':
                    self.emcee_plot(self.fittedValues)

            elif self.inputformat == 'XLSX' and filename.endswith(".xlsx"):
                self.readin_Data_from_xlsx(filename)
                iters = len(self.zarray)
                if self.data_sets is not None:
                    iters = self.data_sets
                for i in range(iters):
                    self.Z = self.zarray[i]
                    self.fittedValues = self.fit_emcee_models(filename)
                    self.process_fitting_results(filename + ' Row' + str(i))
                    if self.LogLevel == 'DEBUG':
                        self.emcee_plot(self.fittedValues)
            elif self.inputformat == 'CSV' and filename.endswith(".csv"):
                self.readin_Data_from_csv(filename)
                iters = len(self.zarray)
                if self.data_sets is not None:
                    iters = self.data_sets
                for i in range(iters):
                    self.Z = self.zarray[i]
                    self.fittedValues = self.fit_emcee_models(filename)
                    self.process_fitting_results(filename + ' Row' + str(i))
                    if self.LogLevel == 'DEBUG':
                        self.emcee_plot(self.fittedValues)
            elif self.inputformat == 'CSV_E4980AL' and filename.endswith(".csv"):
                self.readin_Data_from_csv_E4980AL(filename)
                iters = len(self.zarray)
                if self.data_sets is not None:
                    iters = self.data_sets
                for i in range(iters):
                    self.Z = self.zarray[i]
                    self.fittedValues = self.fit_emcee_models(filename)
                    self.process_fitting_results(filename)
                    if self.LogLevel == 'DEBUG':
                        self.emcee_plot(self.fittedValues)

    def add_lnsigma(self, params):
        if self.lnsigma is not None:
            params.add('__lnsigma', value=self.lnsigma['value'], min=self.lnsigma['min'], max=self.lnsigma['max'])

    def fit_emcee_models(self, filename):
        """
        fit the data to the cole_cole_model first (compensation of the electrode polarization) and then to the defined model.
        """
        params = Parameters()
        if self.model == 'SingleShell':
            params = set_parameters(params, 'single_shell', self.parameters, ep=self.electrode_polarization)
            self.add_lnsigma(params)
            result = self.select_and_solve(self.solvername, single_shell_residual, params, args=(self.omega, self.Z))
        elif self.model == 'DoubleShell':
            params = set_parameters(params, 'double_shell', self.parameters, ep=self.electrode_polarization)
            self.add_lnsigma(params)
            result = self.select_and_solve(self.solvername, double_shell_residual, params, args=(self.omega, self.Z))
        elif self.model == 'ColeCole':
            params = set_parameters(params, 'cole_cole', self.parameters, ep=self.electrode_polarization, ind=self.inductivity, loss=self.high_loss)
            self.add_lnsigma(params)
            result = self.select_and_solve(self.solvername, cole_cole_residual, params, args=(self.omega, self.Z))
        elif self.model == 'ColeColeR':
            params = set_parameters(params, 'cole_cole_r', self.parameters, ep=self.electrode_polarization, ind=self.inductivity, loss=self.high_loss)
            self.add_lnsigma(params)
            result = self.select_and_solve(self.solvername, cole_cole_R_residual, params, args=(self.omega, self.Z))
        elif self.model == 'RC':
            params = set_parameters(params, 'rc', self.parameters, ep=self.electrode_polarization, ind=self.inductivity, loss=self.high_loss)
            self.add_lnsigma(params)
            result = self.select_and_solve(self.solvername, rc_residual, params, args=(self.omega, self.Z))
        elif self.model == 'RC_full':
            params = set_parameters(params, 'RC', self.parameters, ep=self.electrode_polarization, ind=self.inductivity, loss=self.high_loss)
            self.add_lnsigma(params)
            result = self.select_and_solve(self.solvername, RC_residual, params, args=(self.omega, self.Z))
        logger.info(lmfit.fit_report(result))
        logger.debug((result.params.pretty_print()))

        if self.LogLevel == 'DEBUG':
            if self.model == 'SingleShell':
                plot_single_shell(self.omega, self.Z, result, filename)
            elif self.model == 'DoubleShell':
                plot_double_shell(self.omega, self.Z, result, filename)
            elif self.model == 'ColeCole':
                plot_cole_cole(self.omega, self.Z, result, filename)
            elif self.model == 'ColeColeR':
                plot_cole_cole_R(self.omega, self.Z, result, filename)
            elif self.model == 'RC':
                plot_rc(self.omega, self.Z, result, filename)
            elif self.model == 'RC_full':
                plot_RC(self.omega, self.Z, result, filename)
        return result
