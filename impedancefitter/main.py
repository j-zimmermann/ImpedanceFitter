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
import os
import numpy as np
import matplotlib.pyplot as plt
import lmfit

import pandas as pd
import yaml
from copy import deepcopy

from .utils import set_parameters, get_comp_model
from .readin import readin_Data_from_TXT_file, readin_Data_from_collection, readin_Data_from_csv_E4980AL
from .plotting import plot_results, plot_uncertainty

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
        may contain parameters for more than one model.

    Kwargs
    ------

    model: {string, 'SingleShell' OR 'DoubleShell'}
        currently, you can choose between SingleShell and DoubleShell.
    solvername : string
        choose among solvers from lmfit: global (basinhopping, differential_evolution, ...) or local solvers (levenberg-marquardt, nelder-mead, least-squares). See also lmfit documentation.
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
    write_output: {bool, optional}
        decide if you want to dump output to file. Default is False
    fileList: list of strings, optional
        provide a list of files that exclusively should be processed. No other files will be processed.
        This option is particularly good if you have a common fileending of your data (e.g., `.csv`)

    .. todo::
        some are currently not documented


    Attributes
    ----------

    omega: list
        Contains frequencies.
    Z: list
        Contains corresponding impedances.
    protocol: None or string
        Choose 'Iterative' for repeated fits with changing parameter sets, customized approach. If not specified, there is always just one fit for each data set.
    """

    def __init__(self, inputformat, directory=None, **kwargs):
        """
        initializes the Fitter object

        Parameters
        ----------
        modelname: string
            modelname. Must be one of those provided in :func:`impedancefitter.utils.available_models`
        inputformat: string
            must be one of the formats specified in :func:`impedancefitter.utils.available_file_format`

        """
        self.inputformat = inputformat
        self._parse_kwargs(kwargs)

        if directory is None:
            directory = os.getcwd()
        self.directory = directory + '/'
        logger.setLevel(self.LogLevel)

        if not len(logger.handlers):
            # create console handler and set level to debug
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            logger.addHandler(ch)

        if self.inputformat == 'TXT':
            self.prepare_txt()

        self.omega_dict = {}
        self.z_dict = {}
        # read in all data and store it
        if self.fileList is None:
            self.fileList = os.listdir(self.directory)
        for filename in self.fileList:
            filename = os.fsdecode(filename)
            if filename.endswith(self.excludeEnding):
                logger.info("Skipped file {} due to excluded ending.".format(filename))
                continue

            # continues only if data could be read in
            (status, omega, zarray) = self.read_data(filename)
            if status:
                self.omega_dict[str(filename)] = omega
                self.z_dict[str(filename)] = zarray

    def _parse_kwargs(self, kwargs):
        # set defaults
        self.minimumFrequency = None
        self.maximumFrequency = None
        self.LogLevel = 'INFO'
        self.excludeEnding = "impossibleEndingLoL"
        self.solver_kwargs = {}
        self.data_sets = None
        self.current_threshold = None
        self.write_output = False
        self.fileList = None
        self.savefig = False

        # for txt files
        self.trace_b = 'TRACE: B'
        self.skiprows_txt = 21  # header rows inside the *.txt files
        self.skiprows_trace = 2  # line between traces blocks

        # read in kwargs to update defaults
        if 'LogLevel' in kwargs:
            self.LogLevel = kwargs['LogLevel']  # log level: choose info for less verbose output
        if 'minimumFrequency' in kwargs:
            self.minimumFrequency = kwargs['minimumFrequency']
        if 'maximumFrequency' in kwargs:
            self.maximumFrequency = kwargs['maximumFrequency']
        if 'excludeEnding' in kwargs:
            self.excludeEnding = kwargs['excludeEnding']
        if 'data_sets' in kwargs:  # for debugging reasons use for instance only 1 data set
            self.data_sets = kwargs['data_sets']
        if 'current_threshold' in kwargs:
            self.current_threshold = kwargs['current_threshold']
        if 'write_output' in kwargs:
            self.write_output = kwargs['write_output']
        if 'fileList' in kwargs:
            self.fileList = kwargs['fileList']
        if 'trace_b' in kwargs:
            self.trace_b = kwargs['trace_b']
        if 'skiprows_txt' in kwargs:
            self.skiprows_txt = kwargs['skiprows_txt']
        if 'skiprows_trace' in kwargs:
            self.skiprows_trace = kwargs['skiprows_trace']
        if 'savefig' in kwargs:
            self.savefig = kwargs['savefig']

    def initialize_parameters(self, model, parameters):
        """
        The `model_parameters` are initialized either based on a provided `parameterdict` or an input file.

        Parameters
        ----------
        model: string
            model name

        See also
        --------

        :func:`impedancefitter.utils.set_parameters`

        """

        return set_parameters(model, parameterdict=parameters, emcee=self.emcee_tag)

    def initialize_model(self, modelname):
        model = get_comp_model(modelname)
        return model

    def run(self, modelname, solver=None, parameters=None, protocol=None, solver_kwargs={}, modelclass="none"):
        """
        Main function that iterates through all data sets provided.

        Parameters
        ----------

        protocol: string, optional
            Choose 'Iterative' for repeated fits with changing parameter sets, customized approach. If not specified, there is always just one fit for each data set.

        Kwargs
        ------

        solvername: string, optional
            choose an optimizer. Must be available in lmfit. Default is least_squares

        """

        self.modelname = modelname
        self.modelclass = modelclass

        # initialize solver
        if solver is None:
            self.solvername = "least_squares"
        else:
            self.solvername = solver
        if self.solvername == "emcee":
            self.emcee_tag = True
        else:
            self.emcee_tag = False

        self.solver_kwargs = solver_kwargs

        # initialize model
        self.model = self.initialize_model(self.modelname)
        # initialize model parameters
        if parameters is not None:
            logger.debug("Using provided parameter dictionary.")
            assert(isinstance(parameters, dict)), "You need to provide an input dictionary!"
        self.parameters = deepcopy(parameters)

        self.model_parameters = self.initialize_parameters(self.model, self.parameters)
        self.protocol = protocol
        if self.write_output is True:
            open('outfile.yaml', 'w')  # create output file
        for key in self.omega_dict:
            self.omega = self.omega_dict[key]
            self.zarray = self.z_dict[key]
            iters = 1
            # determine number of iterations if more than 1 data set is in file
            if len(self.zarray.shape) > 1:
                self.iters = self.zarray.shape[0]
                logger.debug("Number of data sets:" + str(iters))
            if self.data_sets is not None:
                self.iters = self.data_sets
                logger.debug("Will only iterate over {} data sets.".format(self.iters))
            for i in range(self.iters):
                self.Z = self.zarray[i]
                self.fittedValues = self.process_data_from_file(key, self.model, self.model_parameters, self.modelclass)
                self.process_fitting_results(key + '_' + str(i))
        if self.write_output is True and hasattr(self, "data"):
            outfile = open('outfile.yaml', 'W')  # overwrite output file, or create it, if there is no file
            yaml.dump(self.data, outfile)
        elif not hasattr(self, "data"):
            logger.info("There was no file to process")

    def sequential_run(self, model1, model2, communicate, solver=None, parameters1=None, parameters2=None, modelclass1=None, modelclass2=None, protocol=None):
        """
        Main function that iterates through all data sets provided.

        Parameters
        ----------

        protocol: string, optional
            Choose 'Iterative' for repeated fits with changing parameter sets, customized approach. If not specified, there is always just one fit for each data set.
        """

        # initialize solver
        if solver is None:
            self.solvername = "least_squares"
        else:
            self.solvername = solver
        if self.solvername == "emcee":
            self.emcee_tag = True
        else:
            self.emcee_tag = False

        # initialize model
        self.model1 = self.initialize_model(model1)
        self.model2 = self.initialize_model(model2)

        # initialize model parameters
        if parameters1 is not None:
            logger.debug("Using provided parameter dictionary.")
            assert(isinstance(parameters1, dict)), "You need to provide an input dictionary!"
        self.parameters1 = deepcopy(parameters1)
        # initialize model parameters
        if parameters2 is not None:
            logger.debug("Using provided parameter dictionary.")
            assert(isinstance(parameters2, dict)), "You need to provide an input dictionary!"
        self.parameters2 = deepcopy(parameters2)

        self.model_parameters1 = self.initialize_parameters(self.model1, self.parameters1)
        self.model_parameters2 = self.initialize_parameters(self.model2, self.parameters2)

        self.protocol = protocol
        if self.write_output is True:
            open('outfile.yaml', 'w')  # create output file
        for key in self.omega_dict:
            self.omega = self.omega_dict[key]
            self.zarray = self.z_dict[key]
            iters = 1
            # determine number of iterations if more than 1 data set is in file
            if len(self.zarray.shape) > 1:
                self.iters = self.zarray.shape[0]
                logger.debug("Number of data sets:" + str(iters))
            if self.data_sets is not None:
                self.iters = self.data_sets
                logger.debug("Will only iterate over {} data sets.".format(self.iters))
            for i in range(self.iters):
                self.Z = self.zarray[i]
                self.fittedValues1 = self.process_data_from_file(key, self.model1, self.model_parameters1, modelclass1)
                for c in communicate:
                    try:
                        self.model_parameters2[c].value = self.fittedValues1.best_values[c]
                        self.model_parameters2[c].vary = False
                    except KeyError:
                        print("Key {} you want to communicate is not a valid model key.".format(c))
                self.fittedValues2 = self.process_data_from_file(key, self.model2, self.model_parameters2, modelclass2)
                self.process_sequential_fitting_results(key + '_' + str(i))
        if self.write_output is True and hasattr(self, "data"):
            outfile = open('outfile-sequential.yaml', 'W')  # overwrite output file, or create it, if there is no file
            yaml.dump(self.data, outfile)
        elif not hasattr(self, "data"):
            logger.info("There was no file to process")

    def read_data(self, filename):
        filepath = self.directory + filename
        if self.inputformat == 'TXT' and filename.endswith(".TXT"):
            omega, zarray = readin_Data_from_TXT_file(filepath, self.skiprows_txt, self.skiprows_trace, self.trace_b, self.minimumFrequency, self.maximumFrequency)
        elif self.inputformat == 'XLSX' and filename.endswith(".xlsx"):
            omega, zarray = readin_Data_from_collection(filepath, 'XLSX', minimumFrequency=self.minimumFrequency, maximumFrequency=self.maximumFrequency)
        elif self.inputformat == 'CSV' and filename.endswith(".csv"):
            omega, zarray = readin_Data_from_collection(filepath, 'CSV', minimumFrequency=self.minimumFrequency, maximumFrequency=self.maximumFrequency)
        elif self.inputformat == 'CSV_E4980AL' and filename.endswith(".csv"):
            omega, zarray = readin_Data_from_csv_E4980AL(filepath, minimumFrequency=self.minimumFrequency, maximumFrequency=self.maximumFrequency, current_threshold=self.current_threshold)
        else:
            return False, None, None
        return True, omega, zarray

    def process_fitting_results(self, filename):
        '''
        function writes output into yaml file to prepare statistical analysis of the results.

        Parameters
        ----------
        filename: string
            name of file that is used as key in the output dictionary.
        '''
        if not hasattr(self, "data"):
            self.data = {}
        values = self.fittedValues.best_values
        for key in values:
            values[key] = float(values[key])  # conversion into python native type
        self.data[str(filename)] = values

    def process_sequential_fitting_results(self, filename):
        '''
        function writes output into yaml file to prepare statistical analysis of the results.

        Parameters
        ----------
        filename: string
            name of file that is used as key in the output dictionary.
        '''
        if not hasattr(self, "data"):
            self.data = {}
        values1 = self.fittedValues1.best_values
        values2 = self.fittedValues2.best_values
        for key in values1:
            values1[key] = float(values1[key])  # conversion into python native type
        for key in values2:
            values2[key] = float(values2[key])  # conversion into python native type

        self.data[str(filename)] = {}
        self.data[str(filename)]['model1'] = values1
        self.data[str(filename)]['model2'] = values2

    def model_iterations(self, model):
        r"""
        Information about number of iterations if there is an iterative scheme for a model.

        Note
        ----

        Double Shell model
        k_fit and alpha_fit are the determined values in the cole-cole-fit

        If :attr:`protocol` is equal to `Iterative`, the following procedure is applied:

        #. 1st Fit: the parameters :math:`k` and :math:`\alpha` are fixed(coming from cole-cole-fit), and the data is fitted against the double-shell-model. To be determined in this fit: :math:`\sigma_\mathrm{sup} / k_\mathrm{med}`.

        #. 2nd Fit: the parameters :math:`k`, :math:`\alpha` and :math:`\sigma_\mathrm{sup}` are fixed and the data is fitted again. To be determined in this fit: :math:`\sigma_\mathrm{m}, \varepsilon_\mathrm{m}, \sigma_\mathrm{cp}`.

        #. 3rd Fit: the parameters :math:`k`, :math:`\alpha`, :math:`\sigma_\mathrm{sup}`, :math:`\sigma_\mathrm{m}` and :math:`\varepsilon_\mathrm{m}` are fixed and the data is fitted again. To be determined in this fit: :math:`\sigma_\mathrm{cp}`.

        #. last Fit: the parameters :math:`k`, :math:`\alpha`, :math:`\sigma_\mathrm{sup}, \sigma_\mathrm{ne}, \sigma_\mathrm{np}, \sigma_\mathrm{m}, \varepsilon_\mathrm{m}` are fixed. To be determined in this fit: :math:`\varepsilon_\mathrm{ne}, \sigma_\mathrm{ne}, \varepsilon_\mathrm{np}, \sigma_\mathrm{np}`.


        See also: :func:`impedancefitter.double_shell.double_shell_model`

        """
        iteration_dict = {'DoubleShell': 4,
                          'SingleShell': 2,
                          'ColeCole': 2}
        if model not in iteration_dict:
            logger.info("There exists no iterative scheme for this model.")
            return 1
        else:
            return iteration_dict[model]

    def fix_parameters(self, i, modelname, params, result):
        # since first iteration does not count
        idx = i - 1

        fix_dict = {'DoubleShell': [["kmed", "emed"], ["km", "em"], ["kcp"]],
                    'SingleShell': [["kmed", "emed"]],
                    'ColeCole': [["kdc", "eh"]]}
        fix_list = fix_dict[modelname][idx]
        # fix all parameters to value given in result
        for parameter in fix_list:
            params[parameter].set(vary=False, value=result.best_values[parameter])
        return params

    def fit_data(self, model, parameters, modelclass=None):
        """
        .. todo::
            documentation
        """
        logger.debug('#################################')
        logger.debug('fit data to {} model'.format(model._name))
        logger.debug('#################################')
        # initiate copy of parameters for iterative run
        params = deepcopy(parameters)
        # initiate empty result
        model_result = None

        iters = 1
        if self.protocol == "Iterative":
            try:
                iters = self.model_iterations(modelclass)
            except AttributeError:
                logger.warning("Provide the modelclass kwarg to trigger possible iterative scheme.")
                pass
        for i in range(iters):
            logger.info("###########\nFitting round {}\n###########".format(i + 1))
            if i > 0:
                params = self.fix_parameters(i, modelclass, params, model_result)
            model_result = model.fit(self.Z, params, omega=self.omega, method=self.solvername, fit_kws=self.solver_kwargs)
            logger.info(model_result.fit_report())

            # return solver message (needed since lmfit handles messages
            # differently for the various solvers)
            if hasattr(model_result, "message"):
                if model_result.message is not None:
                    logger.info("Solver message: " + model_result.message)
            if hasattr(model_result, "lmdif_message"):
                if model_result.lmdif_message is not None:
                    logger.info("Solver message (leastsq): " + model_result.lmdif_message)
            if hasattr(model_result, "ampgo_msg"):
                if model_result.ampgo_msg is not None:
                    logger.info("Solver message (ampgo): " + model_result.ampgo_msg)
        return model_result

    def process_data_from_file(self, filename, model, parameters, modelclass=None):
        """
        .. todo::
            documentation
        """
        logger.debug("Going to fit")
        fit_output = self.fit_data(model, parameters, modelclass)
        Z_fit = fit_output.best_fit
        logger.debug("Fit successful")
        if self.LogLevel == 'DEBUG':
            show = True
        # plots if LogLevel is INFO or DEBUG or figure should be saved
        if getattr(logging, self.LogLevel) <= 20 or self.savefig:
            logger.debug("Going to plot results")
            plot_results(self.omega, self.Z, Z_fit, filename, show=show, save=self.savefig)
        return fit_output

    def plot_initial_best_fit(self, sequential=False):
        if not sequential:
            Z_fit = self.fittedValues.best_fit
            Z_init = self.fittedValues.init_fit
            plot_results(self.omega, self.Z, Z_fit, "", show=True, save=False, Z_comp=Z_init)
        else:
            for i in range(2):
                Z_fit = getattr(self, "fittedValues" + str(i + 1)).best_fit
                Z_init = getattr(self, "fittedValues" + str(i + 1)).init_fit
                plot_results(self.omega, self.Z, Z_fit, "", show=True, save=False, Z_comp=Z_init)

    def cluster_emcee_result(self, constant=1e2):
        """
        .. todo::
            documentation
        """
        res = self.fittedValues
        if not self.emcee_tag:
            print("You need to have run emcee as a solver to use this function")
            return
        else:
            lnprob = np.swapaxes(res.lnprob, 0, 1)
            chain = np.swapaxes(res.chain, 0, 1)
            walker_prob = []
            for i in range(lnprob.shape[0]):
                walker_prob.append(-np.mean(lnprob[i]))
            if self.LogLevel == "DEBUG":
                plt.title("walkers sorted by probability")
                plt.xlabel("walker")
                plt.ylabel("negative mean ln probability")
                plt.plot((np.sort(walker_prob)))
                plt.show()
            sorted_indices = np.argsort(walker_prob)
            sorted_fields = np.sort(walker_prob)
            l0 = sorted_fields[0]
            differences = np.diff((sorted_fields))
            if self.LogLevel == "DEBUG":
                plt.title("difference between adjacent walkers")
                plt.xlabel("walker")
                plt.ylabel("difference")
                plt.plot(differences)
                plt.show()
            average_differences = [(x - l0) / (j + 1) for j, x in enumerate((sorted_fields[1::]))]
            if self.LogLevel == "DEBUG":
                plt.title("average difference between current and first walkers")
                plt.ylabel("average difference")
                plt.xlabel("walker")
                plt.plot(average_differences)
                plt.show()
            constant = 1e2

            # set cut to the maximum number of walkers
            cut = len(walker_prob)
            for i in range(differences.size):
                if differences[i] > constant * average_differences[i]:
                    cut = i
                    logger.debug("Cut off at walker {}".format(cut))
                    break
            if self.LogLevel == "DEBUG":
                plt.title("Acceptance fractions after clustering")
                plt.xlabel("walker")
                plt.ylabel("acceptance fraction")
                plt.plot(np.take(res.acceptance_fraction, sorted_indices[:cut:]), label="initially")
                plt.plot(np.take(res.acceptance_fraction, sorted_indices[:cut:]), label="clustered")
                plt.legend()
                plt.show()
            res.new_chain = np.take(chain, sorted_indices[:cut:], axis=0)
            res.new_flatchain = pd.DataFrame(res.new_chain.reshape((-1, res.nvarys)),
                                             columns=res.var_names)

    def emcee_report(self):
        """
        .. todo:: documentation
        """
        if not self.emcee_tag:
            print("You need to have run emcee as a solver to use this function")
            return

        if hasattr(self.fittedValues, 'acor'):
            i = 0
            logger.info("Correlation times per parameter")
            for p in self.fittedValues.params:
                if self.fittedValues.params[p].vary:
                    print(p, self.fittedValues.acor[i])
                    i += 1
        else:
            logger.warning("No autocorrelation data available. Maybe run a longer chain?")

        plt.xlabel("walker")
        plt.ylabel("acceptance fraction")
        plt.plot(self.fittedValues.acceptance_fraction)
        plt.show()

    def plot_uncertainty_interval(self, sigma=1):

        assert isinstance(sigma, int), "Sigma needs to be integer and range between 1 and 3."
        assert sigma >= 1, "Sigma needs to be integer and range between 1 and 3."
        assert sigma <= 3, "Sigma needs to be integer and range between 1 and 3."

        if self.solvername != "emcee":
            print("Not Using emcee")
            ci = self.fittedValues.conf_interval()
        else:
            print("Using emcee")
            ci = self.emcee_conf_interval(self.fittedValues)
        eval1 = lmfit.Parameters()
        eval2 = lmfit.Parameters()
        for p in self.fittedValues.params:
            if p == "__lnsigma":
                continue
            eval1.add(p, value=self.fittedValues.best_values[p])
            eval2.add(p, value=self.fittedValues.best_values[p])
            if p in ci:
                eval1[p].set(value=ci[p][3 - sigma][1])
                eval2[p].set(value=ci[p][3 + sigma][1])

        Z1 = self.fittedValues.eval(params=eval1)
        Z2 = self.fittedValues.eval(params=eval2)
        Z = self.fittedValues.best_fit
        plot_uncertainty(self.fittedValues.userkws['omega'], self.fittedValues.data, Z, Z1, Z2, sigma)

    def emcee_conf_interval(self, result):
        """
        .. todo::
            documentation
        """
        ci = {}
        percentiles = [.5 * (1.0 - 0.9973002039367398),
                       .5 * (1.0 - 0.9544997361036416),
                       .5 * (1.0 - 0.6826894921370859),
                       0.0,
                       .5 * (1.0 + 0.6826894921370859),
                       .5 * (1.0 + 0.9544997361036416),
                       .5 * (1.0 + 0.9973002039367398)]
        pars = [p for p in result.params if result.params[p].vary]
        for i, p in enumerate(pars):
            quantile = np.percentile(result.flatchain[p], np.array(percentiles) * 100)
            ci[p] = list(zip(percentiles, quantile))
        return ci
