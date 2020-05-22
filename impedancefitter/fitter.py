#    The ImpedanceFitter is a package to fit impedance spectra to
#    equivalent-circuit models using open-source software.
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

from .utils import set_parameters, get_equivalent_circuit_model
from .readin import (readin_Data_from_TXT_file,
                     readin_Data_from_collection,
                     readin_Data_from_csv_E4980AL)
from .plotting import plot_impedance, plot_uncertainty

# create logger
logger = logging.getLogger('impedancefitter-logger')


class Fitter(object):
    """
    The main fitting object class.
    All files in the data directory with matching file ending are imported
    to be fitted.

    Parameters
    ----------

    inputformat: string
        The inputformat of the data files. Must be one of the formats specified
        in :func:`impedancefitter.utils.available_file_format`.
        Moreover, the ending of the file must match the `inputformat`.
    directory: string, optional
        Path to data directory.
        Provide the data directory if the data directory is not the
        current working directory.
    LogLevel: {'DEBUG', 'INFO', 'WARNING'}, optional
        choose level for logger. Case DEBUG: the script will output plots after
        each fit, case INFO: the script will output results from each fit
        to the console.
    excludeEnding: string, optional
        For file ending that should be ignored (if there are files with the
        same ending as the chosen inputformat).
        Useful for instance, if there are files like `*_data.csv` and
        `*_result.csv` around and only the first should
        be fitted.
    ending: string, optional
        When `inputformat` is `TXT`, an alternative file ending is possible.
        The default is `.TXT`.
    minimumFrequency: float, optional
        If you want to use another frequency than the minimum frequency
        in the dataset.
    maximumFrequency: float, optional
        If you want to use another frequency than the maximum frequency
        in the dataset.
    data_sets: int, optional
        Use only a certain number of data sets instead of all in directory.
    current_threshold: float, optional
        Use only for data from E4980AL LCR meter to check current.
        If the current is not close to the threshold,
        the data point will be neglected.
    write_output: bool, optional
        Decide if you want to dump output to file. Default is False
    fileList: list of strings, optional
        provide a list of files that exclusively should be processed.
        No other files will be processed.
        This option is particularly good if you have a common
        fileending of your data (e.g., `.csv`)
    savefig: bool, optional
        Decide if you want to save the plots. Default is False.
    trace_b: string, optional
        For TXT files, which contain more than one trace.
        The data is only read in until
        :attr:`trace_b` is found.
        Default is None (then trace_b does not have any effect).
    skiprows_txt: int, optional
        Number of header rows inside a TXT file. Default is 1.
    skiprows_trace: int, optional
        Lines between traces blocks in a TXT file.
        Default is None (then skiprows_trace does not have any effect).
    delimiter: str, optional
        Only for TXT files. If the TXT file is not tab-separated, this
        can be specified here.

    Attributes
    ----------

    omega_dict: dict
        Contains frequency lists that were found in the individual files.
        The keys are the file names, the values the frequencies.
    Z_dict: dict
        Contains corresponding impedances. Note that the values might
        be lists when there was more than one impedance data set in the file.
    fit_data: dict
        Contains the fitting results for each individual file.
        In case of a sequential run, the dictionary contains two
        sub-dictionaries with keys `model1` and `model2` and the results.
    fittedValues: :class:`lmfit.model.ModelResult`
        The fitting result of the last data set that was fitted.
        Exists only when :meth:`run` was called.
    fittedValues1: :class:`lmfit.model.ModelResult`
        The fitting result of the last data set that was fitted.
        Exists only when :meth:`sequential_run` was called and
        corresponds to the first model in this run.
    fittedValues2: :class:`lmfit.model.ModelResult`
        The fitting result of the last data set that was fitted.
        Exists only when :meth:`sequential_run` was called and
        corresponds to the second model in this run.
    """

    def __init__(self, inputformat, directory=None, **kwargs):
        """ initializes the Fitter object
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

        self.omega_dict = {}
        self.z_dict = {}
        # read in all data and store it
        if self.fileList is None:
            self.fileList = os.listdir(self.directory)
        for filename in self.fileList:
            filename = os.fsdecode(filename)
            if filename.endswith(self.excludeEnding):
                logger.info("""Skipped file {}
                             due to excluded ending.""".format(filename))
                continue

            # continues only if data could be read in
            (status, omega, zarray) = self._read_data(filename)
            if status:
                self.omega_dict[str(filename)] = omega
                self.z_dict[str(filename)] = zarray

    def _parse_kwargs(self, kwargs):
        """parses the different kwargs when the Fitter object
        is initialized.
        """

        # set defaults
        self.minimumFrequency = None
        self.maximumFrequency = None
        self.LogLevel = 'INFO'
        self.excludeEnding = "impossibleEndingLoL"
        self.ending = ".TXT"
        self.data_sets = None
        self.current_threshold = None
        self.write_output = False
        self.fileList = None
        self.savefig = False
        self.delimiter = "\t"
        self.emcee_tag = False
        self.protocol = None
        self.solvername = "least_squares"
        self.log = False
        self.solver_kwargs = {}
        self.weighting = None

        # for txt files
        self.trace_b = None
        self.skiprows_txt = 1  # header rows inside the *.txt files
        self.skiprows_trace = None  # line between traces blocks

        # read in kwargs to update defaults
        if 'LogLevel' in kwargs:
            self.LogLevel = kwargs['LogLevel']
        if 'minimumFrequency' in kwargs:
            self.minimumFrequency = kwargs['minimumFrequency']
        if 'maximumFrequency' in kwargs:
            self.maximumFrequency = kwargs['maximumFrequency']
        if 'excludeEnding' in kwargs:
            self.excludeEnding = kwargs['excludeEnding']
        if 'ending' in kwargs:
            self.ending = kwargs['ending']
        if 'data_sets' in kwargs:
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
        if 'delimiter' in kwargs:
            self.delimiter = kwargs['delimiter']

    def visualize_data(self, savefig=False, Zlog=False):
        """Visualize impedance data.

        Parameters
        ----------
        savefig: bool, optional
            Decide if plots should be saved as pdf. Default is False.
        Zlog: bool, optional
            Plot impedance on logscale.
        """

        for key in self.omega_dict:
            zarray = self.z_dict[key]
            if len(zarray.shape) > 1:
                iters = zarray.shape[0]
                logger.debug("Number of data sets to visualise:" + str(iters))
            if self.data_sets is not None:
                iters = self.data_sets
                logger.debug("""Will only iterate
                                over {} data sets.""".format(iters))
            for i in range(iters):
                plot_impedance(self.omega_dict[key], zarray[i],
                               key, save=savefig, Zlog=Zlog)

    def _initialize_parameters(self, model, parameters, emcee=False):
        """
        The `model_parameters` are initialized either based on a
        provided `parameterdict` or an input file.

        Parameters
        ----------
        model: string
            Model name
        parameters: dict
            Parameter dictionary provided by user.
        emcee: bool
            Communicate if emcee was used.

        See also
        --------

        :func:`impedancefitter.utils.set_parameters`

        """

        return set_parameters(model, parameterdict=parameters,
                              emcee=emcee)

    def initialize_model(self, modelname, log):
        """Interface to LMFIT model class.

        The equivalent circuit (represented as a string)
        is parsed and a LMFIT Model is returned.
        This can be useful if one wants to compute the impedance values
        for a given model and use it in a different context.

        Parameters
        ----------

        modelname: string
            Provide equivalent circuit model to be parsed.
        log: bool
            Decide, if logscale is used for the model.

        Returns
        -------

        model: :class:`lmfit.model.Model`
            The resulting LMFIT model.
        """

        model = get_equivalent_circuit_model(modelname, log)
        return model

    def run(self, modelname, solver=None, parameters=None, protocol=None,
            solver_kwargs={}, modelclass="none", log=False, weighting=None):
        """
        Main function that iterates through all data sets provided.

        Parameters
        ----------

        modelname: string
            Name of the model to be parsed. Must be built by those provided in
            :func:`impedancefitter.utils.available_models` and using `+`
            and `parallel(x, y)` as possible representations of series or
            parallel circuit.
        solver: string, optional
            Choose an optimizer. Must be available in LMFIT.
            Default is least_squares
        parameters: dict, optional
            Provide parameters if you do not want
            to read them from a yaml file (for instance in parallel UQ runs).
        protocol: string, optional
            Choose 'Iterative' for repeated fits with changing parameter sets,
            customized approach. If not specified, there is always
            just one fit for each data set.
        solver_kwargs: dict, optional
            Customize the employed solver. Interface to the LMFIT routine.
        modelclass: str, optional
            Pass a modelclass for which the iterative scheme should be used.
            This is experimental support for iterative schemes,
            where parameters can be fixed during the fitting routine.
            In the future, a more intelligent approach could be found.
            See :meth:`impedancefitter.Fitter.model_iterations`
        weighting: str, optional
            Choose a weighting scheme. Default is unit weighting.
            Also possible: proportional weighting. See [Barsoukov2018]_ for more information.

        References
        ----------

        .. [Barsoukov2018] Barsoukov, E., & Macdonald, J. R. (Eds.). (2018).
                           Impedance Spectroscopy: Theory, Experiment, and Applications.
                           (3rd ed.). Hoboken, NJ: John Wiley & Sons, Inc.
                           https://doi.org/10.1002/9781119381860

        """

        self.modelname = modelname
        self.modelclass = modelclass
        self.log = log
        self.weighting = weighting

        # initialize solver
        if solver is not None:
            self.solvername = solver
        if self.solvername == "emcee":
            self.emcee_tag = True

        self.solver_kwargs = solver_kwargs

        # initialize model
        self.model = self.initialize_model(self.modelname, log=self.log)
        # initialize model parameters
        if parameters is not None:
            logger.debug("Using provided parameter dictionary.")
            assert isinstance(parameters, dict), "You need to provide an input dictionary!"
        self.parameters = deepcopy(parameters)

        self.model_parameters = self._initialize_parameters(self.model, self.parameters, self.emcee_tag)
        self.protocol = protocol
        if self.write_output is True:
            open('outfile.yaml', 'w')  # create output file
        for key in self.omega_dict:
            self.omega = self.omega_dict[key]
            self.zarray = self.z_dict[key]
            self.iters = 1
            # determine number of iterations if more than 1 data set is in file
            if len(self.zarray.shape) > 1:
                self.iters = self.zarray.shape[0]
                logger.debug("Number of data sets:" + str(self.iters))
            if self.data_sets is not None:
                self.iters = self.data_sets
                logger.debug("""Will only iterate
                                over {} data sets.""".format(self.iters))
            for i in range(self.iters):
                self.Z = self.zarray[i]
                self.fittedValues = self.process_data_from_file(key,
                                                                self.model,
                                                                self.model_parameters,
                                                                self.modelclass)
                self._process_fitting_results(key + '_' + str(i))
        if self.write_output is True and hasattr(self, "fit_data"):
            outfile = open('outfile.yaml', 'W')
            yaml.dump(self.fit_data, outfile)
        elif not hasattr(self, "fit_data"):
            logger.info("There was no file to process")

    def sequential_run(self, model1, model2, communicate, solver=None,
                       solver_kwargs={}, parameters1=None, parameters2=None,
                       modelclass1=None, modelclass2=None, protocol=None):
        """Main function that iterates through all data sets provided.

        Here, two models are fitted sequentially and fitted parameters can
        be communicated from one model to the other.

        Parameters
        ----------

        model1: string
            Name of first model. Must be built by those provided in
            :func:`impedancefitter.utils.available_models` and using `+`
            and `parallel(x, y)` as possible representations
            of series or parallel circuit
        model2: string
            Name of second model. Must be built by those provided in
            :func:`impedancefitter.utils.available_models` and using `+`
            and `parallel(x, y)` as possible representations
            of series or parallel circuit
        communicate: list of strings
            Names of parameters that should be communicated from model1 to model2.
            Requires that model2 contains a parameter that is named appropriately.
        solver: string, optional
            choose an optimizer. Must be available in LMFIT. Default is least_squares
        solver_kwargs: dict, optional
            Customize the employed solver. Interface to the LMFIT routine.
        parameters1: dict, optional
            Parameters of model1.
            Provide parameters if you do not want to use a yaml file.
        parameters2: dict, optional
            Parameters of model2.
            Provide parameters if you do not want to use a yaml file.
        modelclass1: str, optional
            Pass a modelclass for which the iterative scheme should be used.
            This is experimental support for iterative schemes,
            where parameters can be fixed during the fitting routine.
            In the future, a more intelligent approach could be found.
        modelclass2: str, optional
            Pass a modelclass for which the iterative scheme should be used.
            This is experimental support for iterative schemes,
            where parameters can be fixed during the fitting routine.
            In the future, a more intelligent approach could be found.
        protocol: string, optional
            Choose 'Iterative' for repeated fits with changing parameter sets,
            customized approach. If not specified, there is always just
            one fit for each data set.
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
        self.solver_kwargs = solver_kwargs
        # initialize model
        self.model1 = self.initialize_model(model1, self.log)
        self.model2 = self.initialize_model(model2, self.log)

        # initialize model parameters
        if parameters1 is not None:
            logger.debug("Using provided parameter dictionary.")
            assert isinstance(parameters1, dict), "You need to provide an input dictionary!"
        self.parameters1 = deepcopy(parameters1)
        # initialize model parameters
        if parameters2 is not None:
            logger.debug("Using provided parameter dictionary.")
            assert isinstance(parameters2, dict), "You need to provide an input dictionary!"
        self.parameters2 = deepcopy(parameters2)

        self.model_parameters1 = self._initialize_parameters(self.model1, self.parameters1, self.emcee_tag)
        self.model_parameters2 = self._initialize_parameters(self.model2, self.parameters2, self.emcee_tag)

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
                logger.debug("""Will only iterate
                                over {} data sets.""".format(self.iters))
            for i in range(self.iters):
                self.Z = self.zarray[i]
                self.fittedValues1 = self.process_data_from_file(key,
                                                                 self.model1,
                                                                 self.model_parameters1,
                                                                 modelclass1)
                for c in communicate:
                    try:
                        self.model_parameters2[c].value = self.fittedValues1.best_values[c]
                        self.model_parameters2[c].vary = False
                    except KeyError:
                        logger.error("""Key {} you want to
                                        communicate is not a valid model key.""".format(c))
                        raise
                self.fittedValues2 = self.process_data_from_file(key,
                                                                 self.model2,
                                                                 self.model_parameters2,
                                                                 modelclass2)
                self._process_sequential_fitting_results(key + '_' + str(i))
        if self.write_output is True and hasattr(self, "fit_data"):
            outfile = open('outfile-sequential.yaml', 'W')
            yaml.dump(self.fit_data, outfile)
        elif not hasattr(self, "fit_data"):
            logger.info("There was no file to process")

    def _read_data(self, filename):
        """Read in data from file.

        Parameters
        ----------

        filename: str
            Name of the file to read from.
        """

        filepath = self.directory + filename
        if self.inputformat == 'TXT' and filename.upper().endswith(self.ending.upper()):
            omega, zarray = readin_Data_from_TXT_file(filepath,
                                                      self.skiprows_txt,
                                                      self.skiprows_trace,
                                                      self.trace_b,
                                                      self.delimiter,
                                                      self.minimumFrequency,
                                                      self.maximumFrequency)
        elif self.inputformat == 'XLSX' and filename.upper().endswith(".XLSX"):
            omega, zarray = readin_Data_from_collection(filepath, 'XLSX',
                                                        minimumFrequency=self.minimumFrequency,
                                                        maximumFrequency=self.maximumFrequency)
        elif self.inputformat == 'CSV' and filename.upper().endswith(".CSV"):
            omega, zarray = readin_Data_from_collection(filepath, 'CSV',
                                                        minimumFrequency=self.minimumFrequency,
                                                        maximumFrequency=self.maximumFrequency)
        elif self.inputformat == 'CSV_E4980AL' and filename.endswith(".csv"):
            omega, zarray = readin_Data_from_csv_E4980AL(filepath,
                                                         minimumFrequency=self.minimumFrequency,
                                                         maximumFrequency=self.maximumFrequency,
                                                         current_threshold=self.current_threshold)
        else:
            return False, None, None
        return True, omega, zarray

    def _process_fitting_results(self, filename):
        '''Write output to yaml file to prepare statistical analysis of the results.

        Parameters
        ----------
        filename: string
            name of file that is used as key in the output dictionary.
        '''
        if not hasattr(self, "fit_data"):
            self.fit_data = {}
        values = self.fittedValues.best_values
        for key in values:
            # conversion into python native type
            values[key] = float(values[key])
        self.fit_data[str(filename)] = values

    def _process_sequential_fitting_results(self, filename):
        '''Write output to yaml file to prepare statistical analysis of the results.

        Parameters
        ----------
        filename: string
            name of file that is used as key in the output dictionary.
        '''
        if not hasattr(self, "fit_data"):
            self.fit_data = {}
        values1 = self.fittedValues1.best_values
        values2 = self.fittedValues2.best_values
        # conversion into python native type
        for key in values1:
            values1[key] = float(values1[key])
        for key in values2:
            values2[key] = float(values2[key])

        self.fit_data[str(filename)] = {}
        self.fit_data[str(filename)]['model1'] = values1
        self.fit_data[str(filename)]['model2'] = values2

    def model_iterations(self, modelclass):
        r"""Information about number of iterations
            if there is an iterative scheme for a modelclass.

        Parameters
        ----------

        modelclass: str
            Name of the modelclass. This means that this model is
            represented in the equivalent circuit.

        Returns
        -------
        int
            Number of iteration steps.

        Notes
        -----

        *Double-Shell model*

        The following iterative procedure is applied:

        #. 1st Fit: The data is fitted against a model comprising
           the double-shell model.
           Parameters to be determined in this fitting round: `kmed` and `emed`.

        #. 2nd Fit: The parameters `kmed` and `emed` are fixed and
           the data is fitted again.
           To be determined in this fit: `km` and `em`.

        #. 3rd Fit: In addition, the parameters `km` and `em` are fixed
           and the data is fitted again.
           To be determined in this fit: `kcp`.

        #. last Fit: In addition, he parameter `kcp` is fixed.
           To be determined in this fit: all remaining parameters.

        *Single-Shell model*

        The following iterative procedure is applied:

        #. 1st Fit: The data is fitted against a model comprising
           the single-shell model.
           Parameters to be determined in this fitting round: `kmed` and `emed`.

        #. 2nd Fit: The parameters `kmed` and `emed` are fixed and
           the data is fitted again.

        *Cole-Cole model*

        #. 1st Fit: The data is fitted against a model comprising
           the Cole-Cole model.
           Parameters to be determined in this fitting round: `kdc` and `eh`.

        #. 2nd Fit: The parameters `kdc` and `eh` are fixed and
           the data is fitted again.

        See Also
        --------
        :func:`impedancefitter.double_shell.double_shell_model`
        :func:`impedancefitter.single_shell.single_shell_model`
        :func:`impedancefitter.cole_cole.cole_cole_model`

        """
        iteration_dict = {'DoubleShell': 4,
                          'SingleShell': 2,
                          'ColeCole': 2}
        if modelclass not in iteration_dict:
            logger.info("There exists no iterative scheme for this model.")
            return 1
        else:
            return iteration_dict[modelclass]

    def _fix_parameters(self, i, modelname, params, result):
        """Take parameter value from result and fix the parameter.

        Parameters
        ----------
        i: int
            index of iteration
        params: :class:`lmfit.parameter.Parameters`
            initial parameter dictionary
        result: :class:`lmfit.model.ModelResult`
            results from previous fitting round

        Returns
        -------
        :class:`lmfit.parameter.Parameters`
            updated parameter dictionary

        """
        # since first iteration does not count
        idx = i - 1

        fix_dict = {'DoubleShell': [["kmed", "emed"], ["km", "em"], ["kcp"]],
                    'SingleShell': [["kmed", "emed"]],
                    'ColeCole': [["kdc", "eh"]]}
        fix_list = fix_dict[modelname][idx]
        # fix all parameters to value given in result
        for parameter in result.params:
            params[parameter].set(value=result.best_values[parameter])
        for parameter in fix_list:
            params[parameter].set(vary=False)
        return params

    def _fit_data(self, model, parameters, modelclass=None, weights=None, log=True):
        """Fit data to model.

        Wrapper for LMFIT fitting routine.

        Parameters
        ----------

        model: :class:`lmfit.model.Model` or :class:`lmfit.model.CompositeModel`
            The model to fit to.
        parameters: :class:`lmfit.parameter.Parameters`
            The model parameters to be used.
        modelclass: str, optional
            For an iterative scheme, the modelclass is passed to this function.
        weights: None or :class:`numpy.ndarray`
            Use weights to fit the data. Default is None (no weighting).
            The weights need to have the same shape as the impedance data.

        Returns
        -------

        :class:`lmfit.model.ModelResult`
            Result of fit as LMFIT.ModelResult object.
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
                logger.warning("""Provide the modelclass kwarg
                                  to trigger possible iterative scheme.""")
                pass
        for i in range(iters):
            logger.info("#########\nFitting round {}\n#########".format(i + 1))
            if i > 0:
                params = self._fix_parameters(i, modelclass, params,
                                              model_result)
            if log:
                Z = np.log10(self.Z)
            else:
                Z = self.Z
            model_result = model.fit(Z, params, omega=self.omega,
                                     method=self.solvername,
                                     fit_kws=self.solver_kwargs,
                                     weights=weights)
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

    def process_data_from_file(self, filename, model, parameters,
                               modelclass=None):
        """Fit data from input file to model.

        Wrapper for LMFIT fitting routine.
        If :attr:`LogLevel` is `DEBUG`, the fit result is
        visualised.

        Parameters
        ----------

        filename: str
            Filename, which is contained in the data dictionaries
            :attr:`omega_dict` and :attr:`z_dict`.
        model: :class:`lmfit.model.Model` or :class:`lmfit.model.CompositeModel`
            The model to fit to.
        parameters: :class:`lmfit.parameter.Parameters`
            The model parameters to be used.
        modelclass: str, optional
            For an iterative scheme, the modelclass is passed to this function.

        Returns
        -------

        :class:`lmfit.model.ModelResult`
            Result of fit as :class:`lmfit.model.ModelResult` object.

        """
        logger.debug("Going to fit")
        weights = None
        if self.weighting == "proportional":
            weights = 1. / self.Z.real + 1j / self.Z.imag
        fit_output = self._fit_data(model, parameters, modelclass, log=self.log,
                                    weights=weights)
        if self.log:
            Z_fit = np.power(10, fit_output.best_fit)
        else:
            Z_fit = fit_output.best_fit
        logger.debug("Fit successful")
        if self.LogLevel == 'DEBUG':
            show = True
        # plots if LogLevel is INFO or DEBUG or figure should be saved
        if getattr(logging, self.LogLevel) <= 20 or self.savefig:
            logger.debug("Going to plot results")
            plot_impedance(self.omega, self.Z, filename, Z_fit=Z_fit,
                           show=show, save=self.savefig)
        return fit_output

    def plot_initial_best_fit(self, sequential=False):
        """Plot initial and best fit together.

        This method reveals how good the initial fit was.

        Parameters
        ----------
        sequential: bool, optional
            If a :meth:`sequential_run` was performed, set this value to True.

        """
        if not sequential:
            if self.log:
                Z_fit = np.power(10, self.fittedValues.best_fit)
                Z_init = np.power(10, self.fittedValues.init_fit)
            else:
                Z_fit = self.fittedValues.best_fit
                Z_init = self.fittedValues.init_fit
            plot_impedance(self.omega, self.Z, "", Z_fit=Z_fit,
                           show=True, save=False, Z_comp=Z_init)
        else:
            for i in range(2):
                Z_fit = getattr(self, "fittedValues" + str(i + 1)).best_fit
                Z_init = getattr(self, "fittedValues" + str(i + 1)).init_fit
                if self.log:
                    Z_fit = np.power(10, Z_fit)
                    Z_init = np.power(10, Z_init)
                plot_impedance(self.omega, self.Z, "", Z_fit=Z_fit,
                               show=True, save=False, Z_comp=Z_init)

    def cluster_emcee_result(self, constant=1e2):
        r"""Apply clustering to eliminate low-probability samples.

        Parameters
        ----------

        constant: float
            The constant, which is used to define the threshold
            from which on walkers are eliminated.

        Notes
        -----

        The clustering approach described in [Hou2012]_ is implemented in
        this function.
        The walkers are sorted by probability and subsequently
        the difference between adjacent walker probabilities
        :math:`\Delta_j` is evaluated.
        Then the average difference between the current and the first
        walkeri (:math:`\bar{\Delta}_j`) is evaluated.
        Both differences are compared and a threshold is defined:

        .. math::

            \Delta_j > \mathrm{constant} \cdot \bar{\Delta}_j

        When this inequality becomes true,
        all walkers with :math:`k > j` are thrown away.

        References
        ----------
        .. [Hou2012] Hou, F., Goodman, J., Hogg, D. W., Weare, J., & Schwab, C. (2012).
            An affine-invariant sampler for exoplanet fitting and
            discovery in radial velocity data. Astrophysical Journal, 745(2).
            https://doi.org/10.1088/0004-637X/745/2/198
        """
        res = self.fittedValues
        if not self.emcee_tag:
            print("""You need to have run emcee
                     as a solver to use this function""")
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
            # numerator with j + 1 since enumerate starts from 0
            average_differences = [(x - l0) / (j + 1) for j, x
                                   in enumerate((sorted_fields[1::]))]
            if self.LogLevel == "DEBUG":
                plt.title("average difference between current and first walker")
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
        """Reports acceptance fraction and autocorrelation times.
        """
        if not self.emcee_tag:
            logger.error("""You need to have run emcee
                            as a solver to use this function""")
            return

        if hasattr(self.fittedValues, 'acor'):
            i = 0
            logger.info("Correlation times per parameter")
            for p in self.fittedValues.params:
                if self.fittedValues.params[p].vary:
                    print(p, self.fittedValues.acor[i])
                    i += 1
        else:
            logger.warning("""No autocorrelation data available.
                              Maybe run a longer chain?""")

        plt.xlabel("walker")
        plt.ylabel("acceptance fraction")
        plt.plot(self.fittedValues.acceptance_fraction)
        plt.show()

    def plot_uncertainty_interval(self, sigma=1, sequential=False):
        """Plot uncertainty interval around best fit.

        Parameters
        ----------
        sigma: {1, 2, 3}, optional
            Choose sigma for confidence interval.
        sequential: bool, optional
            Set to True if you performed a sequential run before.

        """

        assert isinstance(sigma, int), "Sigma needs to be integer and range between 1 and 3."
        assert sigma >= 1, "Sigma needs to be integer and range between 1 and 3."
        assert sigma <= 3, "Sigma needs to be integer and range between 1 and 3."

        if sequential:
            iters = 2
            endings = ["1", "2"]
        else:
            iters = 1
            endings = [""]
        for i in range(iters):
            fit_values = getattr(self, "fittedValues{}".format(endings[i]))
            if self.solvername != "emcee":
                logger.debug("Not Using emcee")
                ci = fit_values.conf_interval()
            else:
                logger.debug("Using emcee")
                ci = self.emcee_conf_interval(fit_values)
            eval1 = lmfit.Parameters()
            eval2 = lmfit.Parameters()
            for p in fit_values.params:
                if p == "__lnsigma":
                    continue
                eval1.add(p, value=fit_values.best_values[p])
                eval2.add(p, value=fit_values.best_values[p])
                if p in ci:
                    eval1[p].set(value=ci[p][3 - sigma][1])
                    eval2[p].set(value=ci[p][3 + sigma][1])

            Z1 = fit_values.eval(params=eval1)
            Z2 = fit_values.eval(params=eval2)
            Z = fit_values.best_fit
            plot_uncertainty(fit_values.userkws['omega'], fit_values.data,
                             Z, Z1, Z2, sigma, model=i)

    def emcee_conf_interval(self, result):
        r"""Compute emcee confidence intervals.

        The :math:`1\sigma` to :math:`3\sigma` confidence
        intervals are computed for a fitting result
        generated by emcee since this case is not
        covered by the original LMFIT implementation.

        Parameters
        ----------
        result: :class:`lmfit.model.ModelResult`
            Result from fit.

        Returns
        -------
        dict
            Dictionary containing limits of confidence intervals
            for all free parameters.
            The limits are structured in a list with 7 items, which
            are ordered as follows:

            #. lower limit of :math:`3\sigma` confidence interval.
            #. lower limit of :math:`2\sigma` confidence interval.
            #. lower limit of :math:`1\sigma` confidence interval.
            #. median.
            #. upper limit of  :math:`1\sigma` confidence interval.
            #. upper limit of  :math:`2\sigma` confidence interval.
            #. upper limit of  :math:`3\sigma` confidence interval.

        """
        if not self.emcee_tag:
            print("""You need to have run emcee
                     as a solver to use this function""")
            return

        ci = {}
        percentiles = [.5 * (1.0 - 0.9973002039367398),
                       .5 * (1.0 - 0.9544997361036416),
                       .5 * (1.0 - 0.6826894921370859),
                       .5,
                       .5 * (1.0 + 0.6826894921370859),
                       .5 * (1.0 + 0.9544997361036416),
                       .5 * (1.0 + 0.9973002039367398)]
        pars = [p for p in result.params if result.params[p].vary]
        for i, p in enumerate(pars):
            quantile = np.percentile(result.flatchain[p], np.array(percentiles) * 100)
            ci[p] = list(zip(percentiles, quantile))
        return ci

    def prepare_emcee_run(self):
        """Prepare initial configuration.

           .. todo:: Implement.
        """

    def linkk_test(self, capacitance=False, inductance=False, c=0.85, maxM=100,
                   show=True):
        """Lin-KK test to check Kramers-Kronig validity.

        Parameters
        ----------
        capacitance: bool, optional
            Add extra capacitance to circuit.
        inductance: bool, optional
            Add extra inductance to circuit.
        c: double, optional
            Set threshold for algorithm as described in [Schoenleber2014]_.
            Must be between 0 and 1.
        maxM: int, optional
            Maximum number of RC elements. Default is 100.
        show: bool
            Show plots of test result. Default is True.


       Notes
       -----

       The implementation of the algorithm follows [Schoenleber2014]_
       closely.

       References
       ----------

       .. [Schoenleber2014] Schönleber, M., Klotz, D., & Ivers-Tiffée, E. (2014).
                            A Method for Improving the Robustness of linear Kramers-Kronig Validity Tests.
                            Electrochimica Acta, 131, 20–27. https://doi.org/10.1016/j.electacta.2014.01.034
        """

        results = {}
        for key in self.omega_dict:
            self.omega = self.omega_dict[key]
            self.zarray = self.z_dict[key]
            # determine number of iterations if more than 1 data set is in file
            if len(self.zarray.shape) > 1:
                self.iters = self.zarray.shape[0]
                logger.debug("Number of data sets:" + str(self.iters))
            if self.data_sets is not None:
                self.iters = self.data_sets
                logger.debug("""Will only iterate
                                over {} data sets.""".format(self.iters))
            for i in range(self.iters):
                self.Z = self.zarray[i]
                results[key + str(i)], mus = self._linkk_core(self.omega, self.Z, capacitance,
                                                              inductance, c, maxM)
                if show:
                    Z = results[key + str(i)].best_fit
                    if self.log:
                        Z = np.power(10, Z)
                    plot_impedance(self.omega, self.Z, key + str(i),
                                   Z_fit=Z,
                                   show=True, save=self.savefig, sign=True)

        return results, mus

    def _linkk_core(self, omega, Z, capacitance=False, inductance=False, c=0.85, maxM=100):
        """Core of Lin-KK algorithm.

        Parameters
        ----------
        omega: :class:`numpy.ndarray`, double
            list of frequencies
        Z: :class:`numpy.ndarray`, impedance
            impedance array
        capacitance: bool, optional
            Add extra capacitance to circuit.
        inductance: bool, optional
            Add extra inductance to circuit.
        c: double, optional
            Set threshold for algorithm as described in [Schoenleber2014]_.
            Must be between 0 and 1.
        maxM: int, optional
            Maximum number of RC elements. Default is 100.

        Notes
        -----

        The implementation of the algorithm follows [Schoenleber2014]_
        closely.

        Returns
        -------

        :class:`lmfit.model.ModelResult`
            Result of Lin-KK test
            as :class:`lmfit.model.ModelResult` object.

        """
        # initial value for M
        M = 1

        tau_min = 1. / omega[-1]
        tau_max = 1. / omega[0]
        mu = 1.0

        # initialize initial resistor
        modelstr = "R"
        parameters = {'R': {'value': 1.0}}

        # add capacitance and inductance if specified
        if capacitance:
            modelstr += " + Cstray"
            parameters['C_stray'] = {'value': 1.0}

        if inductance:
            modelstr += " + L"
            parameters['L'] = {'value': 1e-9}
        mus = []
        start = 0
        while mu > c:
            for m in range(M):
                if m >= start:
                    modelstr += " + RCtau_k" + str(m)
                    parameters['k' + str(m) + '_Rk'] = {'value': 1.0}
                    # first parameter is always the same
                    if m == 0:
                        parameters['k0_tauk'] = {'value': tau_min, 'vary': False}
                if M > 1:
                    # note that k - 1 is here m since Python uses zero-based indexing
                    parameters['k' + str(m) + '_tauk'] = {'value': np.power(10, np.log10(tau_min) + m / (M - 1) * np.log10(tau_max / tau_min)),
                                                          'vary': False}
            start = M

            model = self.initialize_model(modelstr, log=self.log)
            model_parameters = self._initialize_parameters(model, parameters)
            model_result = self._fit_data(model, model_parameters, log=self.log)
            mu = _compute_mu(model_result.best_values)
            logger.debug("\nmu = {}, c = {}\n".format(mu, c))
            mus.append(mu)
            if M == maxM:
                logger.warning("Reached maximum number of RC elements.")
                break
            M += 1

        logger.debug("Used M={} RC elements.".format(M - 1))
        return model_result, mus


def _compute_mu(fit_values):
    r"""compute mu from Rk values

    Parameters
    ----------
    fit_values: dict
        Contains all fit values (also fixed taus).
        Rks will be extracted and :math:`\mu` will be computed as
        described in [Schoenleber2014]_.

    Returns
    -------
    double
        Value of :math:`\mu`.
    """

    neg = 0
    pos = 0

    print("fit values:", fit_values)
    for value in fit_values:
        if "_Rk" in value:
            Rk = fit_values[value]
            if Rk < 0:
                neg += -Rk
            else:
                pos += Rk
    return 1. - (neg / pos)
