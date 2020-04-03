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

import lmfit
import yaml
from copy import deepcopy
from .cole_cole import cole_cole_model
from .utils import set_parameters, parameter_names
from .readin import readin_Data_from_TXT_file, readin_Data_from_collection, readin_Data_from_csv_E4980AL
from .plotting import plot_results

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

    def __init__(self, modelname, solvername, inputformat, directory=None, parameters=None, **kwargs):
        """
        initializes the Fitter object

        Parameters
        ----------
        modelname: string
            modelname. Must be one of those provided in :func:`impedancefitter.utils.available_models`
        solvername: string
            choose an optimizer. Must be available in lmfit.
        inputformat: string
            must be one of the formats specified in :func:`impedancefitter.utils.available_file_format`

        """
        self.modelname = modelname
        self.solvername = solvername
        if self.solvername == "emcee":
            self.emcee_tag = True
        else:
            self.emcee_tag = False
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

        if parameters is not None:
            logger.debug("Using provided parameter dictionary.")
            assert(isinstance(parameters, dict)), "You need to provide an input dictionary!"
        self.parameters = deepcopy(parameters)
        if self.inputformat == 'TXT':
            self.prepare_txt()

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
        if 'solver_kwargs' in kwargs:
            self.solver_kwargs = kwargs['solver_kwargs']
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

    def initialize_parameters(self, model):
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

        return set_parameters(model, parameterdict=self.parameters, emcee=self.emcee_tag)

    def initialize_model(self, modelname):
        if modelname == 'ColeCole':
            model = cole_cole_model

        param_names = parameter_names(modelname, ep_cpe=self.electrode_polarization, ind=self.inductivity,
                                      loss=self.high_loss, stray=self.stray)
        return lmfit.Model(model, param_names=param_names, name=modelname)

    def run(self, protocol=None, electrode_polarization=False, inductivity=False, high_loss=False, stray_capacitance=False):
        """
        Main function that iterates through all data sets provided.

        Parameters
        ----------

        protocol: string, optional
            Choose 'Iterative' for repeated fits with changing parameter sets, customized approach. If not specified, there is always just one fit for each data set.
        electrode_polarization: bool
            Switch on whether to account for electrode polarization or not. Currently, only a CPE correction is possible.
        inductivity: bool
            Switch on whether to account for cable inductance. See :func:`impedancefitter.elements.Z_in`.
        high_loss: bool
            Switch on whether to account for cable inductance AND capacitance. See :func:`impedancefitter.elements.Z_loss`.
        stray_capacitance: bool
            Switch on whether to account for stray capacitance
        """

        # initialize global correction parameters
        self.electrode_polarization = electrode_polarization
        self.inductivity = inductivity
        self.high_loss = high_loss
        self.stray = stray_capacitance

        # initialize model
        self.model = self.initialize_model(self.modelname)
        # initialize model parameters
        self.model_parameters = self.initialize_parameters(self.model)
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
            self.read_data(filename)
            # determine number of iterations if more than 1 data set is in file
            iters = len(self.zarray)
            if self.data_sets is not None:
                iters = self.data_sets
            for i in range(iters):
                self.Z = self.zarray[i]
                self.fittedValues = self.process_data_from_file(filename, self.model, self.model_parameters)
                self.process_fitting_results(filename + '_' + str(i))

    def read_data(self, filename):
        filepath = self.directory + filename
        if self.inputformat == 'TXT' and filename.endswith(".TXT"):
            self.omega, self.zarray = readin_Data_from_TXT_file(filepath, self.skiprows_txt, self.skiprows_trace, self.trace_b, self.minimumFrequency, self.maximumFrequency)
        elif self.inputformat == 'XLSX' and filename.endswith(".xlsx"):
            self.omega, self.zarray = readin_Data_from_collection(filepath, 'XLSX', minimumFrequency=self.minimumFrequency, maximumFrequency=self.maximumFrequency)
        elif self.inputformat == 'CSV' and filename.endswith(".csv"):
            self.omega, self.zarray = readin_Data_from_collection(filepath, 'CSV', minimumFrequency=self.minimumFrequency, maximumFrequency=self.maximumFrequency)
        elif self.inputformat == 'CSV_E4980AL' and filename.endswith(".csv"):
            self.omega, self.zarray = readin_Data_from_csv_E4980AL(filepath, minimumFrequency=self.minimumFrequency, maximumFrequency=self.maximumFrequency, current_threshold=self.current_threshold)

    def process_fitting_results(self, filename):
        '''
        function writes output into yaml file to prepare statistical analysis of the results.

        Parameters
        ----------
        filename: string
            name of file that is used as key in the output dictionary.
        '''
        values = dict(self.fittedValues.params.valuesdict())
        for key in values:
            values[key] = float(values[key])  # conversion into python native type
        self.data = {str(filename): values}
        if self.write_output is True:
            outfile = open('outfile.yaml', 'a')  # append to the already existing file, or create it, if ther is no file
            yaml.dump(self.data, outfile)

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

    def fit_data(self, model, parameters):
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
            iters = self.model_iterations(self.model._name)
        for i in range(iters):
            logger.info("###########\nFitting round {}\n###########".format(i + 1))
            if i > 0:
                params = self.fix_parameters(i, model._name, params, model_result)
            model_result = self.model.fit(self.Z, params, omega=self.omega, method=self.solvername, fit_kws=self.solver_kwargs)
            logger.info(model_result.fit_report())
            if self.solvername != "ampgo":
                logger.info("Solver message: " + model_result.message)
            else:
                logger.info("Solver message: " + model_result.ampgo_msg)
        return model_result

    def process_data_from_file(self, filename, model, parameters):
        """
        .. todo::
            documentation
        """
        fit_output = self.fit_data(model, parameters)
        Z_fit = fit_output.best_fit
        if self.LogLevel == 'DEBUG':
            show = True
        # plots if LogLevel is INFO or DEBUG or figure should be saved
        if getattr(logging, self.LogLevel) <= 20 or self.savefig:
            plot_results(self.omega, self.Z, Z_fit, filename, show=show, save=self.savefig)
        return fit_output
