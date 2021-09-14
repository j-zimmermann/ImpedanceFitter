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

import os
import numpy as np
import matplotlib.pyplot as plt
import lmfit

import pandas as pd
import yaml
from copy import deepcopy

from .utils import set_parameters, get_equivalent_circuit_model, get_labels, _return_resistance_capacitance
from .readin import (readin_Data_from_TXT_file,
                     readin_Data_from_collection,
                     readin_Data_from_csv_E4980AL,
                     readin_Data_from_dta)
from .plotting import plot_impedance, plot_uncertainty, plot_bode, plot_resistance_capacitance
from . import set_logger
from .variance import weighting_residual, variance_estimate

import logging
logger = logging.getLogger(__name__)


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
    voltage_threshold: float, optional
        Use only for data from E4980AL LCR meter to check voltage.
        If the voltage is not close to the threshold,
        the data point will be neglected.
    E4980AL_tolerance: float, optional
        Set tolerance for `current_threshold` and/or `voltage_threshold`.
    write_output: bool, optional
        Decide if you want to dump output to file. Default is False
    fileList: list of strings, optional
        provide a list of files that exclusively should be processed.
        No other files will be processed.
        This option is particularly good if you have a common
        fileending of your data (e.g., `.csv`)
    show: bool, optional
        Decide if you want to see the plots during the fitting procedure.
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
        Only for TXT and CSV files. If the TXT/CSV file is not tab-separated, this
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

        set_logger(self.LogLevel)

        self.omega_dict = {}
        self.z_dict = {}
        # read in all data and store it
        if self.fileList is None:
            self.fileList = os.listdir(self.directory)
        read_data_sets = 0
        for filename in self.fileList:
            if self.data_sets:
                if read_data_sets == self.data_sets:
                    logger.debug("Reached maximum number of data sets.")
                    break
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
                read_data_sets += 1

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
        self.voltage_threshold = None
        self.E4980AL_tolerance = 1e-2
        self.write_output = False
        self.fileList = None
        self.savefig = False
        if self.inputformat == "TXT":
            self.delimiter = "\t"
        elif self.inputformat == "CSV":
            self.delimiter = None
        self.emcee_tag = False
        self.solvername = "least_squares"
        self.log = False
        self.eps = False
        self.weighting_model = False
        self.solver_kwargs = {}
        self.weighting = None
        self.show = False
        self.savemodelresult = True
        self.report = False

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
        if 'voltage_threshold' in kwargs:
            self.voltage_threshold = kwargs['voltage_threshold']
        if 'E4980AL_tolerance' in kwargs:
            self.E4980AL_tolerance = kwargs['E4980AL_tolerance']
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
        if 'show' in kwargs:
            self.show = kwargs['show']
        if 'savefig' in kwargs:
            self.savefig = kwargs['savefig']
        if 'delimiter' in kwargs:
            self.delimiter = kwargs['delimiter']

    def visualize_data(self, savefig=False, Zlog=False,
                       allinone=False, plottype="impedance",
                       show=True, legend=True):
        """Visualize impedance data.

        Parameters
        ----------
        savefig: bool, optional
            Decide if plots should be saved as pdf. Default is False.
            Saves the file as `filename` + index of dataset + `_impedance_overview.pdf`.
            If `allinone` is True, then it is `allinone_impedance_overview.pdf`.
        Zlog: bool, optional
            Plot impedance on logscale.
        show: bool, optional
            Decide if you want to immediately see the plot.
        allinone: bool, optional
            Visualize all data sets in one plot
        plottype: str, optional
            Choose between standard impedance plot ('impedance'), resistance / capacitance ("RC") and bode plot ('bode').
        legend: str, optional
            Choose if a legend should be shown. Recommended to switch to False
            when using large datasets.
        """
        if not savefig and not show:
            logger.warning("""visualize_data does not have any effect if you
                              neither save nor show the plot.""")
            return
        totaliters = 0
        savefigtmp = savefig
        showtmp = show
        labels = ["Data", None, None]

        for key in self.z_dict:
            zarray = self.z_dict[key]
            totaliters += zarray.shape[0]

        for key in self.omega_dict:
            append = False
            zarray = self.z_dict[key]
            iters = zarray.shape[0]
            logger.debug("Number of data sets in file {} to visualise: {}".format(key, iters))

            for i in range(iters):
                title = key + str(i)
                if allinone and totaliters > 1:
                    append = True
                    savefigtmp = False
                    labels = [key + str(i), None, None]
                    showtmp = False
                elif allinone and totaliters == 1:
                    labels = [key + str(i), None, None]
                    savefigtmp = savefig
                    title = "allinone"
                    showtmp = show

                if plottype == "impedance":
                    plot_impedance(self.omega_dict[key], zarray[i], title=title, show=showtmp,
                                   save=savefigtmp, Zlog=Zlog, append=append, labels=labels,
                                   legend=legend)
                elif plottype == "bode":
                    plot_bode(self.omega_dict[key], zarray[i], title=title, show=showtmp,
                              save=savefigtmp, append=append, labels=labels,
                              legend=legend)
                elif plottype == "RC":
                    plot_resistance_capacitance(self.omega_dict[key], zarray[i],
                                                title=title, show=showtmp,
                                                save=savefigtmp, append=append, labels=labels,
                                                legend=legend)

                else:
                    raise RuntimeError("You chose an invalid plottype")
                totaliters -= 1

    def _initialize_parameters(self, model, parameters, emcee=False, weighting_model=False):
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
                              emcee=emcee, weighting_model=weighting_model)

    def initialize_model(self, modelname, log=False, eps=False):
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
        diel: bool
            Convert to complex permittivity and fit this instead of impedance.


        Returns
        -------

        model: :class:`lmfit.model.Model`
            The resulting LMFIT model.
        """

        model = get_equivalent_circuit_model(modelname, log, eps)
        return model

    def run(self, modelname, solver=None, parameters=None,
            solver_kwargs={}, log=False, weighting=None,
            show=False, report=False, savemodelresult=True, eps=False,
            residual="parts", limits_residual=None, weighting_model=False):
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
        solver_kwargs: dict, optional
            Customize the employed solver. Interface to the LMFIT routine.
        weighting: str, optional
            Choose a weighting scheme. Default is unit weighting.
            Also possible: proportional and modulus weighting. See [Barsoukov2018]_ and [Orazem2017]_ for more information.
            Moreover, weighted least squares is possible. This should only be chosen if the data can be averaged.
            The average data will be fitted and the standard deviation is used for the weights.
            The keyword for this is WLS. In this case, you need to provide weights.
        savemodelresult: bool, optional
            Saves all :class:`lmfit.model.ModelResult` instances to plot or evaluate uncertainty later.
        residual: str
            Plot relative difference w.r.t. real and imaginary part if `parts`.
            Plot relative difference w.r.t. absolute value if `absolute`.
            Plot difference (residual) if `diff`.
        limits_residual: list, optional
            List with entries `[bottom, top]` for y-axis of residual plot.

        """

        self.modelname = modelname
        self.log = log
        self.eps = eps
        self.weighting_model = weighting_model
        if weighting is not None:
            if isinstance(weighting, str) and weighting in ["proportional", "modulus", "reweighting", "WLS"]:
                self.weighting = weighting
            elif self.weighting_model:
                raise RuntimeError("""The variable `weighting` must be None if you want to use the weighting_model. Otherwise, you must set weighting_model to False.""")
            else:
                raise RuntimeError("""The variable `weighting` must be a string and refer to an available weighting scheme.
                                  Use either `proportional` or `modulus` or `WLS`.""")
        else:
            self.weighting = weighting
        self.show = show
        self.report = report

        # initialize solver
        if solver is not None:
            self.solvername = solver
        else:
            self.solvername = "least_squares"
        SCALAR_METHODS = ['nelder', 'Nelder-Mead', 'powell', 'Powell',
                          'cg', 'CG', 'bfgs', 'BFGS', 'newton', 'Newton-CG',
                          'lbfgsb', 'L-BFGS-B', 'tnc', 'TNC', 'cobyla', 'COBYLA',
                          'slsqp', 'SLSQP', 'dogleg', 'trust-ncg', 'differential_evolution',
                          'trust-constr', 'trust-exact', 'trust-krylov']

        if weighting_model and self.solvername not in SCALAR_METHODS:
            raise AttributeError("Provide a scalar method. They are {}".format(SCALAR_METHODS))
        if self.solvername == "emcee":
            self.emcee_tag = True
        else:
            self.emcee_tag = False

        self.solver_kwargs = solver_kwargs

        # initialize model
        self.model = self.initialize_model(self.modelname, log=self.log, eps=self.eps)
        # initialize model parameters
        if parameters is not None:
            logger.debug("Using provided parameter dictionary.")
            assert isinstance(parameters, dict), "You need to provide an input dictionary!"
        self.parameters = deepcopy(parameters)

        self.model_parameters = self._initialize_parameters(self.model, self.parameters, self.emcee_tag, self.weighting_model)
        if self.write_output is True:
            open('outfile.yaml', 'w')  # create output file
        # TODO:
        # implement WLS here
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
                self.fittedValues = self.process_data_from_file(key + str(i),
                                                                self.model,
                                                                self.model_parameters,
                                                                residual,
                                                                limits_residual,
                                                                self.weighting_model)
                self._process_fitting_results(key + '_' + str(i))
        if self.write_output is True and hasattr(self, "fit_data"):
            outfile = open('outfile.yaml', 'W')
            yaml.dump(self.fit_data, outfile)
        elif not hasattr(self, "fit_data"):
            logger.info("There was no file to process")

    def sequential_run(self, model1, model2, communicate, solver=None,
                       solver_kwargs={}, parameters1=None, parameters2=None,
                       weighting=None):
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
        weighting: str, optional
            Choose a weighting scheme. Default is unit weighting.
            Also possible: proportional weighting. See [Barsoukov2018]_ for more information.
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
        self.model1 = self.initialize_model(model1, self.log, self.eps)
        self.model2 = self.initialize_model(model2, self.log, self.eps)

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
                self.fittedValues1 = self.process_data_from_file(key + str(i),
                                                                 self.model1,
                                                                 self.model_parameters1)
                for c in communicate:
                    try:
                        self.model_parameters2[c].value = self.fittedValues1.best_values[c]
                        self.model_parameters2[c].vary = False
                    except KeyError:
                        logger.error("""Key {} you want to
                                        communicate is not a valid model key.""".format(c))
                        raise
                self.fittedValues2 = self.process_data_from_file(key + str(i),
                                                                 self.model2,
                                                                 self.model_parameters2)
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
        elif self.inputformat == 'DTA' and filename.upper().endswith(".DTA"):
            omega, zarray = readin_Data_from_dta(filepath,
                                                 minimumFrequency=self.minimumFrequency,
                                                 maximumFrequency=self.maximumFrequency)
        elif self.inputformat == 'XLSX' and filename.upper().endswith(".XLSX"):
            omega, zarray = readin_Data_from_collection(filepath, 'XLSX',
                                                        minimumFrequency=self.minimumFrequency,
                                                        maximumFrequency=self.maximumFrequency)
        elif self.inputformat == 'CSV' and filename.upper().endswith(".CSV"):
            omega, zarray = readin_Data_from_collection(filepath, 'CSV',
                                                        delimiter=self.delimiter,
                                                        minimumFrequency=self.minimumFrequency,
                                                        maximumFrequency=self.maximumFrequency)
        elif self.inputformat == 'CSV_E4980AL' and filename.endswith(".csv"):
            omega, zarray = readin_Data_from_csv_E4980AL(filepath,
                                                         minimumFrequency=self.minimumFrequency,
                                                         maximumFrequency=self.maximumFrequency,
                                                         current_threshold=self.current_threshold,
                                                         voltage_threshold=self.voltage_threshold,
                                                         tolerance=self.E4980AL_tolerance)
        else:
            return False, None, None

        if zarray.size == 0:
            raise RuntimeError("""In file {} an empty data set was provided.
                                  If you used the E4980AL and set a current_threshold,
                                  the reason might be that there was no data point that matched
                                  the current_threshold.""".format(filename))

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
            if key == "Zdata":
                continue
            # conversion into python native type
            values[key] = float(values[key])
        self.fit_data[str(filename)] = values
        if self.savemodelresult:
            if not hasattr(self, "model_results"):
                self.model_results = {}
            self.model_results[str(filename)] = self.fittedValues

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
        if self.savemodelresult:
            if not hasattr(self, "model_results"):
                self.model_results = {}
            self.model_results[str(filename)] = {}
            self.model_results[str(filename)]['model1'] = self.fittedValues1
            self.model_results[str(filename)]['model2'] = self.fittedValues2

    def _fit_data(self, model, parameters, weights=None,
                  log=True, eps=False, weighting_model=False):
        """Fit data to model.

        Wrapper for LMFIT fitting routine.

        Parameters
        ----------

        model: :class:`lmfit.model.Model` or :class:`lmfit.model.CompositeModel`
            The model to fit to.
        parameters: :class:`lmfit.parameter.Parameters`
            The model parameters to be used.
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
        # initiate copy of parameters
        # TODO: still needed?
        params = deepcopy(parameters)
        # initiate empty result
        model_result = None

        logger.debug("#########\nFitting started \n#########")
        if log:
            Z = np.log10(self.Z)
        elif eps:
            Z = 1. / (1j * self.omega * params['c0all'] * self.Z)
        else:
            Z = self.Z
        max_nfev = None
        if 'max_nfev' in self.solver_kwargs:
            max_nfev = self.solver_kwargs['max_nfev']
            tmp_dict = {key: self.solver_kwargs[key] for key in set(list(self.solver_kwargs.keys())) - set(['max_nfev'])}
        else:
            tmp_dict = self.solver_kwargs
        if weighting_model:
            model_result = lmfit.minimize(weighting_residual, params, method=self.solvername, args=(self.omega,), kws={'Zdata': Z, 'model': model})
            best_values = deepcopy(model_result.params.valuesdict())
            setattr(model_result, "best_values", best_values)
        else:
            # this is also k=0 in reweighting
            model_result = model.fit(Z, params, omega=self.omega,
                                     method=self.solvername,
                                     fit_kws=tmp_dict,
                                     weights=weights,
                                     max_nfev=max_nfev)
            if self.weighting == "reweighting":
                # raise NotImplementedError("Not yet implemented")
                tol = 1e-7
                tolA = 1
                tolPhi = 1
                tolZ = 1
                varA_old = 0
                varPhi_old = 0
                kmax = 50000
                k = 0
                while (np.greater_equal(tolA, tol) or np.greater_equal(tolZ, tol) or np.greater_equal(tolPhi, tol)) and k < kmax:
                    if k == 0:
                        Zfit = model_result.best_fit
                        varA, varPhi = variance_estimate(Z, Zfit)
                        fitparams = np.fromiter(model_result.params.values(), dtype=float)
                    varA_old, varPhi_old = varA, varPhi
                    fitparamsold = fitparams
                    weights = 1 / (np.abs(Z)**2 * varA_old) + 1j / (np.abs(Z)**2 * (varA_old + varPhi_old) * np.angle(Z)**2)
                    model_result = model.fit(Z, params, omega=self.omega,
                                             method=self.solvername,
                                             fit_kws=tmp_dict,
                                             weights=weights,
                                             max_nfev=max_nfev)
                    Zfit = model_result.best_fit
                    params = model_result.params
                    fitparams = np.fromiter(model_result.params.values(), dtype=float)
                    varA, varPhi = variance_estimate(Z, Zfit)
                    tolA = np.abs(varA_old - varA) / varA
                    tolPhi = np.abs(varPhi_old - varPhi) / varPhi
                    diff = (fitparamsold - fitparams)
                    tolZ = np.sqrt(diff.dot(diff)) / np.sqrt(fitparams.dot(fitparams))
                    print(tolA, tolPhi, tolZ)
                    k += 1
                logger.info("Converged after {} iterations".format(k))

        if self.weighting in ["proportional", "modulus", "WLS"]:
            _calculate_statistics(model_result)
        if not self.report:
            if weighting_model:
                logger.debug(lmfit.fit_report(model_result))
            else:
                logger.debug(model_result.fit_report())
        else:
            if weighting_model:
                logger.info(lmfit.fit_report(model_result))
            else:
                logger.info(model_result.fit_report())

        # return solver message (needed since lmfit handles messages
        # differently for the various solvers)
        if hasattr(model_result, "message"):
            if model_result.message is not None:
                logger.debug("Solver message: " + model_result.message)
        if hasattr(model_result, "lmdif_message"):
            if model_result.lmdif_message is not None:
                logger.debug("Solver message (leastsq): " + model_result.lmdif_message)
        if hasattr(model_result, "ampgo_msg"):
            if model_result.ampgo_msg is not None:
                logger.debug("Solver message (ampgo): " + model_result.ampgo_msg)
        return model_result

    def get_resistance_capacitance(self):
        self.R_dict = {}
        self.C_dict = {}
        for key in self.omega_dict:
            self.omega = self.omega_dict[key]
            self.zarray = self.z_dict[key]
            rarray = np.zeros(self.zarray.shape, dtype=np.float128)
            carray = np.zeros(self.zarray.shape, dtype=np.float128)
            self.iters = 1
            # determine number of iterations if more than 1 data set is in file
            if len(self.zarray.shape) > 1:
                self.iters = self.zarray.shape[0]
                logger.debug("Number of data sets:" + str(self.iters))
            for i in range(self.iters):
                self.Z = self.zarray[i]
                rarray[i], carray[i] = _return_resistance_capacitance(self.omega, self.Z)
            self.R_dict[key] = rarray
            self.C_dict[key] = carray

    def get_admittance(self):
        self.Y_dict = {}
        for key in self.omega_dict:
            self.zarray = self.z_dict[key]
            xarray = np.zeros(self.zarray.shape, dtype=np.complex128)
            self.iters = 1
            # determine number of iterations if more than 1 data set is in file
            if len(self.zarray.shape) > 1:
                self.iters = self.zarray.shape[0]
                logger.debug("Number of data sets:" + str(self.iters))
            for i in range(self.iters):
                self.Z = self.zarray[i]
                xarray[i] = (1. / self.Z)
            self.Y_dict[key] = xarray

    def process_data_from_file(self, filename, model, parameters,
                               residual="parts", limits_residual=None,
                               weighting_model=False):
        """Fit data from input file to model.

        Wrapper for LMFIT fitting routine.

        Parameters
        ----------

        filename: str
            Filename, which is contained in the data dictionaries
            :attr:`omega_dict` and :attr:`z_dict`.
        model: :class:`lmfit.model.Model` or :class:`lmfit.model.CompositeModel`
            The model to fit to.
        parameters: :class:`lmfit.parameter.Parameters`
            The model parameters to be used.
        residual: str
            Plot relative difference w.r.t. real and imaginary part if `parts`.
            Plot relative difference w.r.t. absolute value if `absolute`.
            Plot difference (residual) if `diff`.
        limits_residual: list, optional
            List with entries `[bottom, top]` for y-axis of residual plot.


        Returns
        -------

        :class:`lmfit.model.ModelResult`
            Result of fit as :class:`lmfit.model.ModelResult` object.

        """
        logger.debug("Going to fit")
        weights = None
        if self.weighting == "proportional":
            weights = 1. / self.Z.real + 1j / self.Z.imag
        elif self.weighting == "modulus":
            weights = 1. / np.abs(self.Z) + 1j / np.abs(self.Z)
        elif self.weighting == "WLS":
            weights = self.Zstd
        fit_output = self._fit_data(model, parameters, log=self.log,
                                    eps=self.eps, weights=weights, weighting_model=weighting_model)
        if self.log:
            Z_fit = np.power(10, fit_output.best_fit)
        elif self.eps:
            Z_fit = 1. / (1j * self.omega * fit_output.best_fit * parameters['c0all'])
        elif self.weighting_model:
            tmp_model = get_equivalent_circuit_model(self.modelname)
            Z_fit = tmp_model.eval(omega=self.omega, **fit_output.best_values)
        else:
            Z_fit = fit_output.best_fit
        logger.debug("Fit successful")
        # plots if LogLevel is DEBUG or figure should be saved
        if self.show or self.savefig:
            if self.show:
                logger.info("Going to plot results")
            if self.savefig:
                logger.info("Going to save plot of fit result to file.")
            title = "fit_result_" + filename
            plot_impedance(self.omega, self.Z, title=title, Z_fit=Z_fit,
                           show=self.show, save=self.savefig, residual=residual,
                           limits_residual=limits_residual)
        return fit_output

    def plot_initial_best_fit(self, sequential=False, save=False, show=True):
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
            plot_impedance(self.omega, self.Z, "Comparison of initial and best fit", Z_fit=Z_fit,
                           show=show, save=save, Z_comp=Z_init)
        else:
            for i in range(2):
                Z_fit = getattr(self, "fittedValues" + str(i + 1)).best_fit
                Z_init = getattr(self, "fittedValues" + str(i + 1)).init_fit
                if self.log:
                    Z_fit = np.power(10, Z_fit)
                    Z_init = np.power(10, Z_init)
                plot_impedance(self.omega, self.Z, "", Z_fit=Z_fit,
                               show=show, save=save, Z_comp=Z_init)

    def cluster_emcee_result(self, constant=1e2, show=False):
        r"""Apply clustering to eliminate low-probability samples.

        Parameters
        ----------

        constant: float
            The constant, which is used to define the threshold
            from which on walkers are eliminated.
        show: bool, optional
            Plot clustering result.

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

        assert hasattr(self, "model_results"), "You need to have saved the LMFIT model results."
        if not self.emcee_tag:
            print("""You need to have run emcee
                     as a solver to use this function""")
            return
        else:
            for fits in self.model_results:
                res = self.model_results[fits]
                lnprob = np.swapaxes(res.lnprob, 0, 1)
                chain = np.swapaxes(res.chain, 0, 1)
                walker_prob = []
                for i in range(lnprob.shape[0]):
                    walker_prob.append(-np.mean(lnprob[i]))
                if show:
                    plt.title("walkers sorted by probability")
                    plt.xlabel("walker")
                    plt.ylabel("negative mean ln probability")
                    plt.plot((np.sort(walker_prob)))
                    plt.show()
                sorted_indices = np.argsort(walker_prob)
                sorted_fields = np.sort(walker_prob)
                l0 = sorted_fields[0]
                differences = np.diff((sorted_fields))
                if show:
                    plt.title("difference between adjacent walkers")
                    plt.xlabel("walker")
                    plt.ylabel("difference")
                    plt.plot(differences)
                    plt.show()
                # numerator with j + 1 since enumerate starts from 0
                average_differences = [(x - l0) / (j + 1) for j, x
                                       in enumerate((sorted_fields[1::]))]
                if show:
                    plt.title("average difference between current and first walker")
                    plt.ylabel("average difference")
                    plt.xlabel("walker")
                    plt.plot(average_differences)
                    plt.show()

                # set cut to the maximum number of walkers
                cut = len(walker_prob)
                for i in range(differences.size):
                    if differences[i] > constant * average_differences[i]:
                        cut = i
                        logger.debug("Cut off at walker {}".format(cut))
                        break
                if show:
                    plt.title("Acceptance fractions after clustering")
                    plt.xlabel("walker")
                    plt.ylabel("acceptance fraction")
                    plt.plot(np.take(res.acceptance_fraction, sorted_indices[::]), label="initial")
                    plt.plot(np.take(res.acceptance_fraction, sorted_indices[:cut:]), label="clustered")
                    plt.legend()
                    plt.show()
                setattr(res, "new_chain", np.take(chain, sorted_indices[:cut:], axis=0))
                setattr(res, "new_flatchain", pd.DataFrame(res.new_chain.reshape((-1, res.nvarys)),
                                                           columns=res.var_names))

    def emcee_report(self):
        """Reports acceptance fraction and autocorrelation times.
        """
        if not self.emcee_tag:
            logger.error("""You need to have run emcee
                            as a solver to use this function""")
            return

        assert hasattr(self, "model_results"), "You need to have saved the LMFIT model results."
        for fits in self.model_results:
            fittedValues = self.model_results[fits]
            if hasattr(fittedValues, 'acor'):
                i = 0
                logger.info("Correlation times per parameter")
                for p in fittedValues.params:
                    if fittedValues.params[p].vary:
                        print(p, fittedValues.acor[i])
                        i += 1
            else:
                logger.warning("""No autocorrelation data available.
                                  Maybe run a longer chain?""")

            plt.xlabel("walker")
            plt.ylabel("acceptance fraction")
            plt.plot(fittedValues.acceptance_fraction)
            plt.show()

            highest_prob = np.argmax(fittedValues.lnprob)
            hp_loc = np.unravel_index(highest_prob, fittedValues.lnprob.shape)
            mle_soln = fittedValues.chain[hp_loc]
            paramsmle = {}
            for i, par in enumerate(fittedValues.var_names):
                paramsmle[par] = mle_soln[i]

            print('\nMaximum Likelihood Estimation from emcee       ')
            print('-------------------------------------------------')
            print('Parameter  MLE Value   Median Value   Uncertainty')
            fmt = '  {:5s}  {:11.5f} {:11.5f}   {:11.5f}'.format
            for name in fittedValues.var_names:
                print(fmt(name, paramsmle[name], fittedValues.params[name].value,
                          fittedValues.params[name].stderr))

    def plot_emcee_chains(self, title=True, savefig=False):
        for fits in self.model_results:
            fittedValues = self.model_results[fits]
            ndim = len(fittedValues.var_names)
            fig, axes = plt.subplots(ndim, figsize=(15, 10), sharex=True)
            samples = fittedValues.chain
            labels = get_labels(fittedValues.var_names)
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels[fittedValues.var_names[i]])
                ax.yaxis.set_label_coords(-0.1, 0.5)

            axes[-1].set_xlabel("step number")
            plt.suptitle('Chains ' + fits, y=1.05)
            plt.tight_layout()
            if savefig:
                plt.savefig(fits + "__chains.pdf")
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
        assert hasattr(self, "model_results"), "You need to have saved the LMFIT model results."

        for d in self.fit_data:
            if sequential:
                iters = 2
                endings = ["1", "2"]
            else:
                iters = 1
                endings = [""]
            for i in range(iters):
                modelresult = self.model_results[d]
                if sequential:
                    modelresult = modelresult['model' + endings[i]]
                if self.solvername != "emcee":
                    logger.debug("Not Using emcee")
                    ci = modelresult.conf_interval()
                else:
                    logger.debug("Using emcee")
                    ci = self.emcee_conf_interval(modelresult)
                eval1 = lmfit.Parameters()
                eval2 = lmfit.Parameters()
                for p in modelresult.params:
                    if p == "__lnsigma":
                        continue
                    eval1.add(p, value=modelresult.best_values[p])
                    eval2.add(p, value=modelresult.best_values[p])
                    if p in ci:
                        eval1[p].set(value=ci[p][3 - sigma][1])
                        eval2[p].set(value=ci[p][3 + sigma][1])

                Z1 = modelresult.eval(params=eval1)
                Z2 = modelresult.eval(params=eval2)
                Z = modelresult.best_fit
                plot_uncertainty(modelresult.userkws['omega'], modelresult.data,
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

    def prepare_emcee_run(self, leastsquaresresult,
                          lnsigma={'value': np.log(0.1), 'min': np.log(0.001), 'max': np.log(2)},
                          nwalkers=100,
                          radius=1e-4, weighted=False,
                          burn=500,
                          steps=10e3,
                          thin=10,
                          fix_parameters=[]):
        """Prepare initial configuration based on previous least squares run.


        Parameters
        ----------

        leastsquaresresult: :class:`lmfit.ModelResult`
            Result of previous least squares run on same model.
            Basically the initial setting. Could also stem from
            a previous MCMC run.
        lnsigma: dict
            Value of initial guess for experimental uncertainty.
        nwalkers: int
            Number of walkers.
        radius: float
            Radius of ball around initial guess.
        burn: int
            Number of steps to be removed from MCMC chain in burn-in phase.
        steps: int
            Length of MCMC chain.
        thin: int
            Take only every `thin`-th step into account.
        weighted: bool
            If `False`, `lnsigma` will be used.
        fix_parameters: list
            Tell emcee to fix certain parameters to their least squares result.

        Notes
        -----
        The result from a previous least squares run is taking and
        the emcee run is prepared.
        The walkers are created and their initial positions are assigned.
        The initial positions are placed in a tight Gaussian ball around
        the least squares guess. Their radius can be set manually.
        Important parameters for the MCMC chains are set.

        Returns
        -------

        dict
            dictionary, which can be passed to run method via `solver_kwargs` keyword.
        """
        # take results from least squares
        parameters_dict = {}
        for l in leastsquaresresult.params:
            parameters_dict[l] = {}
            parameters_dict[l]['value'] = leastsquaresresult.params[l].value
            parameters_dict[l]['min'] = -np.inf
            parameters_dict[l]['max'] = np.inf
            if l in fix_parameters:
                parameters_dict[l]['vary'] = False
            else:
                parameters_dict[l]['vary'] = leastsquaresresult.params[l].vary

        # get parameters in right order
        parameters = []
        for p in leastsquaresresult.var_names:
            if parameters_dict[p]["vary"] is True:
                parameters.append(p)

        # for__lnsigma if lnsigma has not been fitted yet
        if '__lnsigma' not in parameters:
            parameters_dict['__lnsigma'] = lnsigma

        ls_values = [parameters_dict[p]['value'] for p in parameters]
        nvarys = len(parameters)
        if '__lnsigma' not in parameters:
            ls_values.append(lnsigma['value'])
            nvarys += 1
        # and assemble tiny gaussian ball around least squares result
        pos = (np.array(ls_values) * (1. + radius * np.random.randn(nwalkers, nvarys)))
        solver_kwargs = {'seed': 42,
                         # 'nan_policy': 'omit',
                         'burn': burn,
                         'steps': steps,
                         'nwalkers': nwalkers,
                         'thin': thin,
                         'pos': pos,
                         'is_weighted': weighted}
        return solver_kwargs, parameters_dict

    def linkk_test(self, capacitance=False, inductance=False,
                   c=0.85, maxM=100, show=True, limits=[-2, 2], weighting="modulus"):
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
        limits: list, optional
            Lower and upper limit of residual.
        weighting: "modulus" or None
            Apply modulus weighting (as in [Schoenleber2014]_) or process
            unweighted data.

        Returns
        -------

        results : dict
            Values of Lin-KK test for each file.
        mus: dict
            All `mu` values during Lin-KK run for each file.
        residuals: dict
            Least-squares residuals during Lin-KK run for each file.

        Notes
        -----

        The implementation of the algorithm follows [Schoenleber2014]_
        closely.

        If the option `savefig` is generally enabled, the plot result
        of the LinKK-Test will be saved to a pdf-file.

        References
        ----------

        .. [Schoenleber2014] Schönleber, M., Klotz, D., & Ivers-Tiffée, E. (2014).
                             A Method for Improving the Robustness of linear Kramers-Kronig Validity Tests.
                             Electrochimica Acta, 131, 20–27. https://doi.org/10.1016/j.electacta.2014.01.034
        """

        results = {}
        mus = {}
        residuals = {}
        titlebegin = "Lin-KK test "

        if capacitance:
            titlebegin += "capacitance "
        if inductance:
            titlebegin += "inductance "
        for key in self.omega_dict:
            self.omega = self.omega_dict[key]
            self.zarray = self.z_dict[key]
            if self.omega.size < maxM:
                logger.warning("""
                                  The maximal number of RC elements is greater than
                                  the number of frequencies ({}). This does not make sense.
                                  Consider decreasing the maximal number of RC elements if
                                  the LinKK test uses more RC elements than frequencies.""".format(self.omega.size))
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
                results[key + str(i)], mus[key + str(i)], residuals[key + str(i)] = (
                    self._linkk_core(self.omega, self.Z, capacitance,
                                     inductance, c, maxM, weighting=weighting))
                if show or self.savefig:
                    Z_fit = self._get_linkk_impedance(results[key + str(i)])
                    plot_impedance(self.omega, self.Z, title=titlebegin + str(key) + str(i),
                                   Z_fit=Z_fit, residual="absolute", limits_residual=limits,
                                   show=show, save=self.savefig, sign=True)

        return results, mus, residuals

    def _get_linkk_impedance(self, params):
        """Compute Lin-KK impedance.

        Parameters
        ----------

        params: dict
            Parameter returned from Lin-KK test.


        Returns
        -------
        :class:`numpy.ndarray`, complex
            impedances

        """

        modelR = "R"
        R = self.initialize_model(modelR)
        Z_fit = R.eval(omega=self.omega, R=params['R'])

        start = 1

        if 'C' in params:
            modelC = "C"
            C = self.initialize_model(modelC)
            Z_fit = np.sum([Z_fit, C.eval(omega=self.omega, C=params['C'])], axis=0)
            start += 1

        if 'L' in params:
            modelL = "L"
            L = self.initialize_model(modelL)
            Z_fit = np.sum([Z_fit, L.eval(omega=self.omega, L=params['L'])], axis=0)
            start += 1

        M = int((len(params) - start) / 2)

        modelRC = "RCtau"
        RC = self.initialize_model(modelRC)

        RCtaus = np.array([RC.eval(omega=self.omega, Rk=params['R_' + str(m)],
                           tauk=params['tau_' + str(m)]) for m in range(M)])
        if M > 1:
            add = np.sum(RCtaus, axis=0)
            Z_fit = np.sum([Z_fit, add], axis=0)
        else:
            Z_fit = np.sum([Z_fit, RCtaus[0]], axis=0)
        return Z_fit

    def _linkk_core(self, omega, Z, capacitance=False, inductance=False, c=0.85, maxM=100,
                    weighting="modulus"):
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
        weighting: "modulus" or None
            Apply modulus weighting (as in [Schoenleber2014]_) or process
            unweighted data.


        Notes
        -----

        The implementation of the algorithm follows [Schoenleber2014]_
        closely.

        Returns
        -------

        fitresult : dict
            Values of Lin-KK test.
        mus: list
            All `mu` values during Lin-KK run.
        residuals: list
            Least-squares residuals during Lin-KK run.

        """
        # initial value for M
        M = 1

        tau_min = 1. / omega[-1]
        tau_max = 1. / omega[0]
        mu = 1.0

        # initialize initial resistor
        modelR = "R"
        R = self.initialize_model(modelR)

        # initialize matrix for linear least-squares

        if weighting == "modulus":
            weight = 1. / np.abs(self.Z)
        elif weighting is None:
            weight = np.ones(self.Z.size)
            print(weight)
        else:
            raise RuntimeError("This is not a valid weighting option.")

        weightM = np.diag(weight)
        Abase = R.eval(omega=self.omega, R=1.0).real
        start = 1
        # add capacitance and inductance if specified
        if capacitance:
            modelC = "C"
            C = self.initialize_model(modelC)
            Abase = np.c_[Abase, C.eval(omega=self.omega, C=1.0)]
            start += 1

        if inductance:
            modelL = "L"
            L = self.initialize_model(modelL)
            Abase = np.c_[Abase, L.eval(omega=self.omega, L=1.0)]
            start += 1
        mus = []
        res = []

        modelRC = "RCtau"
        tauk = tau_max

        RC = self.initialize_model(modelRC)

        while mu > c:
            A = np.copy(Abase)
            if M > 1:
                for m in range(M):
                    tauk = np.power(10, np.log10(tau_min) + m / (M - 1) * np.log10(tau_max / tau_min))
                    A = np.c_[A, RC.eval(omega=self.omega, Rk=1.0, tauk=tauk)]
            else:
                A = np.c_[A, RC.eval(omega=self.omega, Rk=1.0, tauk=tauk)]

            # solve least-squares problem for real and imaginary part together
            real = weightM.dot(A.real)
            imag = weightM.dot(A.imag)
            Zweighted = weightM.dot(self.Z)

            a = np.concatenate((real, imag))
            y = np.concatenate((Zweighted.real, Zweighted.imag))
            # note that rcond=-1 means that singular values are hardly reached
            b, residualslstsq, rank, s = np.linalg.lstsq(a, y, rcond=-1)

            rks = b[start:]

            mu = _compute_mu(rks)

            # when M is greater than the number of frequencies,
            # the residual is not computed by numpy thus we put a -1 there
            if len(residualslstsq) == 1:
                res.append(residualslstsq[0])
            else:
                print(residualslstsq)
                logger.warning("""
                               No residual was computed for either of 2 reasons:
                               1.) M is greater than number of frequencies.
                               2.) Singular values were detected and removed.
                               Will add -1 to list of residuals.""")
                res.append(-1)

            mus.append(mu)

            # Note by Julius: to show that code worked, I computed the residual as given in Schönleber paper
            # Then, I compared to the lstsq residual computed by numpy.
            # for documentation reasons, I kept the lines commented.
            """
            Z_fit = R.eval(omega=omega, R=b[0])

            if capacitance and not inductance:
                Z_fit = np.sum([Z_fit, C.eval(omega=omega, C=1. / b[1])], axis=0)
            elif capacitance and inductance:
                Z_fit = np.sum([Z_fit, C.eval(omega=omega, C=1. / b[1])], axis=0)
                Z_fit = np.sum([Z_fit, L.eval(omega=omega, L=b[2])], axis=0)
            elif inductance and not capacitance:
                Z_fit = np.sum([Z_fit, L.eval(omega=omega, L=b[1])], axis=0)

            if M > 1:
                taus = np.array([np.power(10, np.log10(tau_min) + m / (M - 1) * np.log10(tau_max / tau_min)) for m in range(M)])
                RCtaus = np.array([RC.eval(omega=self.omega, Rk=rks[m], tauk=taus[m]) for m in range(M)])
                add = np.sum(RCtaus, axis=0)
                Z_fit = np.sum([Z_fit, add], axis=0)

            elif M == 1:
                add = RC.eval(omega=self.omega, Rk=rks[0], tauk=tauk)
                Z_fit = np.sum([Z_fit, add], axis=0)
            rescomp = np.sum(((Z_fit.real - self.Z.real) * weight)**2 + ((Z_fit.imag - self.Z.imag) * weight)**2)
            if not np.isclose(res[-1], rescomp):
                print("Detected difference between lstlsq and residual at M=", M, res[-1], rescomp)
            """

            M += 1
            if M > maxM:
                logger.warning("Reached maximum number of RC elements.")
                break

        logger.debug("\nmu = {}, c = {}\n".format(mu, c))

        fitresult = {'R': b[0]}

        # Found negative inductance and / or capacitance values when fitting.
        # I am not sure if this is correct.

        if capacitance and not inductance:
            fitresult['C'] = 1. / b[1]
            if b[1] < 0:
                logger.warning("The LinKK-fitted capacitance is negative!")
        elif capacitance and inductance:
            fitresult['C'] = 1. / b[1]
            if b[1] < 0:
                logger.warning("The LinKK-fitted capacitance is negative!")
            fitresult['L'] = b[2]
            if b[2] < 0:
                logger.warning("The LinKK-fitted inductance is negative!")
        elif inductance and not capacitance:
            fitresult['L'] = b[1]
            if b[1] < 0:
                logger.warning("The LinKK-fitted inductance is negative!")

        M -= 1
        if M > 1:
            taus = np.array([np.power(10, np.log10(tau_min) + m / (M - 1) * np.log10(tau_max / tau_min)) for m in range(M)])
            for m in range(M):
                fitresult['R_' + str(m)] = rks[m]
                fitresult['tau_' + str(m)] = taus[m]
        elif M == 1:
            fitresult['R_' + str(0)] = rks[0]
            fitresult['tau_' + str(0)] = tau_max

        logger.info("Used M={} RC elements.".format(M))
        return fitresult, mus, res


def _compute_mu(fit_values):
    r"""compute mu from Rk values

    Parameters
    ----------
    fit_values: list
        Contains Rk fit values.
        Rks are used to compute :math:`\mu` as
        described in [Schoenleber2014]_.

    Returns
    -------
    double
        Value of :math:`\mu`.
    """

    neg = 0
    pos = 0

    for value in fit_values:
        if value < 0:
            neg += -value
        else:
            pos += value

    if np.greater(pos, 0):
        mu = 1. - (neg / pos)
    else:
        mu = -np.inf
    return mu


def _calculate_statistics(model_result):
    """Calculate the fitting statistics.

    Function taken from LMFIT source code
    and modified
    https://github.com/lmfit/lmfit-py/blob/05bbc07321480d02dcd13c122ba16d42ec3d4eae/lmfit/minimizer.py

    It corrects for the weighting and thus makes fits comparable.
    """
    model_result.residual = (model_result.best_fit - model_result.data).ravel().view(np.float)
    if isinstance(model_result.residual, np.ndarray):
        model_result.chisqr = (model_result.residual**2).sum()
    else:
        model_result.chisqr = model_result.residual
    model_result.redchi = model_result.chisqr / max(1, model_result.nfree)
    # this is -2*loglikelihood
    model_result.chisqr = max(model_result.chisqr, 1.e-250 * model_result.ndata)
    _neg2_log_likel = model_result.ndata * np.log(model_result.chisqr / model_result.ndata)
    model_result.aic = _neg2_log_likel + 2 * model_result.nvarys
    model_result.bic = _neg2_log_likel + np.log(model_result.ndata) * model_result.nvarys
