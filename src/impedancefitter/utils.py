#    The ImpedanceFitter is a package to fit impedance spectra to
#    equivalent-circuit models using open-source software.
#
#    Copyright (C) 2018, 2019 Leonard Thiele, leonard.thiele[AT]uni-rostock.de
#    Copyright (C) 2018, 2019, 2020 Julius Zimmermann,
#                                   julius.zimmermann[AT]uni-rostock.de
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
import re
from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd
import pyparsing as pp
import schemdraw
import schemdraw.elements as elm
import yaml
from lmfit import CompositeModel, Model
from schemdraw.util import Point
from scipy.constants import epsilon_0 as e0
from scipy.integrate import simpson

from .cole_cole import (
    cole_cole_2_model,
    cole_cole_2tissue_model,
    cole_cole_3_model,
    cole_cole_4_model,
    cole_cole_model,
    cole_cole_R_model,
    havriliak_negami,
    havriliak_negamitissue,
    raicu,
)
from .cpe import (
    cpe_ct_model,
    cpe_ct_tissue_model,
    cpe_ct_w_model,
    cpe_model,
    cpe_onset_model,
    cpetissue_model,
)
from .double_shell import double_shell_bh_model, double_shell_model
from .double_shell_ellipsoid import double_shell_ellipsoid_model
from .double_shell_wall import double_shell_wall_bh_model, double_shell_wall_model
from .double_shell_wall_ellipsoid import double_shell_wall_ellipsoid_model
from .ecis import Z_ECIS_Lo_Ferrier
from .elements import (
    Z_C,
    Z_L,
    Z_R,
    Z_ADIa_a,
    Z_ADIa_r,
    Z_ADIb_a,
    Z_ADIb_r,
    Z_ADII_a,
    Z_ADII_r,
    Z_stray,
    Z_w,
    Z_wo,
    Z_ws,
    eps,
    log,
    parallel,
)
from .loss import Z_in, Z_loss, Z_skin
from .particle_suspension import particle_bh_model, particle_model
from .randles import Z_randles, Z_randles_CPE
from .RC import RC_model, drc_model, rc_model, rc_tau_model
from .single_shell import single_shell_bh_model, single_shell_model
from .single_shell_ellipsoid import single_shell_ellipsoid_model
from .single_shell_wall import single_shell_wall_bh_model, single_shell_wall_model
from .single_shell_wall_ellipsoid import single_shell_wall_ellipsoid_model

logger = logging.getLogger(__name__)


def convert_diel_properties_to_impedance(omega, eps_r, sigma, c0):
    r"""Return impedance from dielectric properties.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`, double
        frequency array
    eps_r: :class:`numpy.ndarray`, double
        relative permittivity
    sigma: :class:`numpy.ndarray`, double
        conductivity in S/m
    c0: double
        unit capacitance of device

    Returns
    -------
    :class:`numpy.ndarray`, complex
        impedance array

    Notes
    -----
    Use that the impedance is

    .. math::

        Z = (j \omega \varepsilon_\mathrm{r}^\ast c_0)^{-1} ,

    where :math:`\varepsilon_\mathrm{r}^\ast` is the relative
    complex permittivity (see for instance [Grant1958]_
    for further explanation). Note that the vacuum permittivity
    :math:`\varepsilon_0` is contained in :math:`c_0`.

    In the function, the variable `epsc` describes the term

    .. math::

        \omega \varepsilon_\mathrm{r}^\ast

    """
    epsc = omega * eps_r - 1j * sigma / e0
    return 1.0 / (1j * epsc * c0)


def return_diel_properties(omega, Z, c0):
    r"""Return relative permittivity and conductivity
    from impedance spectrum in cavity with known unit capacitance.

    Notes
    -----
    Use that the impedance is

    .. math::

        Z = (j \omega \varepsilon_\mathrm{r}^\ast c_0)^{-1} ,

    where :math:`\varepsilon_\mathrm{r}^\ast` is the relative
    complex permittivity (see for instance [Grant1958]_
    for further explanation). Note that the vacuum permittivity
    :math:`\varepsilon_0` is contained in :math:`c_0`.

    When the unit capacitance :math:`c_0` of the device is known,
    a direct mapping from impedance to
    relative complex permittivity is possible:

    .. math ::

        \varepsilon_\mathrm{r}^\ast = (j \omega Z c_0)^{-1} =
        \frac{\varepsilon^\ast}{\varepsilon_0}

    The unit capacitance (or air capacitance) of the device is defined as

    .. math::

        c_0 = \frac{\varepsilon_0 A}{d}

    for a parallel-plate capacitor with electrode area `A` and spacing `d` but can
    also be measured in a calibration step.

    The relative permittivity is the real part of :math:`\varepsilon_\mathrm{r}^\ast`
    and the conductivity is the negative imaginary part times the frequency and the
    vacuum permittivity.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`, double
        frequency array
    Z: :class:`numpy.ndarray`, complex
        impedance array
    c0: double
        unit capacitance of device

    Returns
    -------
    eps_r: :class:`numpy.ndarray`, double
        relative permittivity
    conductivity: :class:`numpy.ndarray`, double
        conductivity in S/m

    References
    ----------
    .. [Grant1958] Grant, F. A. (1958).
                   Use of complex conductivity in the
                   representation of dielectric phenomena.
                   Journal of Applied Physics, 29(1), 76–80.
                   https://doi.org/10.1063/1.1722949
    """
    epsc = 1.0 / (1j * omega * Z * c0)
    eps_r = epsc.real
    conductivity = -epsc.imag * e0 * omega
    return eps_r, conductivity


def return_dielectric_modulus(omega, Z, c0):
    r"""Return dielectric modulus.

    Notes
    -----
    The dielectric modulus is  :math:`M = 1 / \varepsilon_\mathrm{r}^\ast`.
    See [Bordi2001]_ for further explanation.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`, double
        frequency array
    Z: :class:`numpy.ndarray`, complex
        impedance array
    c0: double
        unit capacitance of device

    Returns
    -------
    ReM: :class:`numpy.ndarray`, double
        real part of modulus
    ImM: :class:`numpy.ndarray`, double
        imaginary part of modulus

    References
    ----------
    .. [Bordi2001] Bordi, F., & Cametti, C. (2001).
                   Occurrence of an intermediate relaxation process
                   in water-in-oil microemulsions below percolation:
                   The electrical modulus formalism.
                   Journal of Colloid and Interface Science, 237(2), 224–229.
                   https://doi.org/10.1006/jcis.2001.7456
    """
    epsc = 1.0 / (1j * omega * Z * c0)
    M = 1.0 / epsc
    ReM = M.real
    ImM = M.imag
    return ReM, ImM


# ruff: noqa: C901
def check_parameters(bufdict):
    """Check parameters for physical correctness.

    Parameters
    ----------
    bufdict: dict
        Contains all parameters and their values

    Notes
    -----
    All parameters are forced to be greater or equal zero.
    There are only two exceptions.
    """
    # capacitances in pF
    capacitancespF = ["c0", "Cs"]
    capacitances = ["C"]
    # taus in ns
    taus = ["tau", "tauE", "tau1", "tau2", "tau3", "tau4"]
    zerotoones = ["p", "a", "alpha", "beta", "gamma", "a1", "a2", "a3", "a4", "nu"]
    permittivities = ["em", "ecp", "emed", "ene", "enp", "epsl", "eps", "epsinf"]

    # __lnsigma and Rk (Lin-KK test)
    # can be negative and do not need to be checked
    exceptions = ["__lnsigma", "Rk"]
    for p in bufdict:
        if p in exceptions:
            continue

        tmp = p.split("_")
        if len(tmp) == 2:
            par = tmp[1]
        elif len(tmp) == 1:
            par = tmp[0]
        else:
            raise RuntimeError(
                f"The parameter {p} cannot be split in prefix and suffix."
            )

        if par in exceptions:
            continue

        if par in capacitancespF:
            assert not np.isclose(
                bufdict[p].value, 0.0, atol=1e-5
            ), f"""{p} is used in pF, do you really want it to be that small?
                   It will be ignored in the analysis"""

        if par in zerotoones:
            if not (0 <= bufdict[p].value <= 1.0):
                raise ValueError(
                    f"{p} is an exponent or the volume fraction "
                    "that needs to be between 0.0 and 1.0."
                    " Change your initial value."
                )
            if bufdict[p].vary:
                if bufdict[p].min < 0.0 or bufdict[p].min > 1.0:
                    logger.debug(
                        f"{p} is an exponent that needs to be between"
                        " 0.0 and 1.0. Changed your min value to 0."
                    )
                    bufdict[p].set(min=0.0)
                if bufdict[p].max > 1.0:
                    logger.debug(
                        f"{p} is an exponent that needs to be between 0.0 and 1.0."
                        "Changed your max value to 1.0."
                    )
                    bufdict[p].set(max=1.0)
            continue

        if par in taus:
            assert not np.isclose(
                bufdict[p].value, 0.0, atol=1e-7
            ), "tau is used in ns, do you really want it to be that small?"

        # check permittivities
        if par in permittivities:
            assert (
                bufdict[p].value >= 1.0
            ), f"""The permittivity {p} needs to be greater than
                   or equal to 1. Change the initial value."""
            if bufdict[p].vary:
                if bufdict[p].min < 1.0:
                    logger.debug(
                        f"""The permittivity {p} needs to be greater
                                   than or equal to 1. Changed the min value to 1."""
                    )
                    bufdict[p].set(min=1.0)
                if bufdict[p].max < 1.0:
                    logger.debug(
                        f"""The permittivity {p} needs to be greater than
                                   or equal to 1. Changed the max value to inf."""
                    )
                    bufdict[p].set(max=np.inf)
            continue

        if bufdict[p].vary:
            if bufdict[p].min < 0.0:
                logger.debug(f"{p} needs to be positive. Changed your min value to 0.")
                bufdict[p].set(min=0.0)
                if par in capacitances:
                    # to avoid division by zero
                    bufdict[p].set(min=1e-12)
            if bufdict[p].max < 0.0:
                logger.debug(
                    f"{p} needs to be positive. Changed your max value to inf."
                )
                bufdict[p].set(max=np.inf)
    return bufdict


# ruff: noqa: C901
def set_parameters(model, parameterdict=None, emcee=False, weighting_model=False):
    """
    Parameters
    ----------
    model: :py:class:`lmfit.model.Model`
        The LMFIT model used for fitting.
    parameterdict: dict, optional
        A dictionary containing parameters for model
        with *min*, *max*, *vary* info for LMFIT.
        If it is None (default), the parameters are read in from a yaml-file.
    emcee: bool, optional
        if emcee is used, an additional `__lnsigma` parameter will be set
    weighting_model: bool, optional
        if a weighting model is used, the variance will be fit as well


    Returns
    -------
    params: :py:class:`lmfit.parameter.Parameters`
        LMFIT Parameters object.
    """
    if parameterdict is None:
        try:
            infile = open("input.yaml")
            bufdict = yaml.safe_load(infile)
        except OSError:
            logger.error("Please provide a yaml-input file.")
            raise
    else:
        try:
            bufdict = parameterdict.copy()
        except KeyError:
            logger.error(
                "Your parameterdict lacks an entry for the model."
                f" Required are: {model.param_names}"
            )
            raise

    bufdict = _clean_parameters(bufdict, model.param_names)
    logger.debug(f"Setting values for parameters {model.param_names}")
    logger.debug(f"Parameters: {bufdict}")
    for key in model.param_names:
        model.set_param_hint(key, **bufdict[key])
    parameters = model.make_params()
    parameters = check_parameters(parameters)
    if emcee and "__lnsigma" not in parameterdict:
        logger.warning(
            "You did not provide the parameter '__lnsigma'. "
            "It is needed for the emcee of unweighted data (as implemented here). "
            "We now use the lmfit default:"
            "value=np.log(0.1), min=np.log(0.001), max=np.log(2)"
        )
        parameters.add(
            "__lnsigma", value=np.log(0.1), min=np.log(0.001), max=np.log(2.0)
        )
    elif emcee and "__lnsigma" in parameterdict:
        parameters.add("__lnsigma", **parameterdict["__lnsigma"])

    if weighting_model:
        if "stdA" not in parameterdict and "stdPhi" not in parameterdict:
            raise RuntimeError(
                "You need to provide the variables stdA and stdPhi "
                "if you want to use a weighting model."
            )
        for var in ["stdA", "stdPhi"]:
            if "min" in parameterdict[var]:
                if parameterdict[var]["min"] < 0:
                    logger.warning(
                        f"You set the minimum value of parameter {var}"
                        "to a negative value. "
                        "That does not work and the value is set to 0."
                    )
                    parameterdict[var]["min"] = 0
            else:
                parameterdict[var]["min"] = 0
            if "max" in parameterdict[var]:
                if parameterdict[var]["max"] < 0:
                    logger.warning(
                        f"You set the maximum value of parameter {var}"
                        "to a negative value. "
                        "That does not work and the value is set to inf."
                    )
                parameterdict[var]["max"] = np.inf
            else:
                parameterdict[var]["min"] = 0
            parameters.add(var, **parameterdict[var])
    return parameters


def _clean_parameters(params, names):
    """
    clean parameter dicts that are passed to the fitter.
    get rid of parameters that are not needed.

    Parameters
    ----------
    params: dict
        input dictionary
    modelName: string
        name of model corresponding to the ones listed
    names: list of strings
        names of model parameters
    """
    for p in list(params.keys()):
        if p not in names:
            del params[p]

    assert Counter(names) == Counter(
        params.keys()
    ), "You need to provide the following parameters (maybe with prefixes)" + str(names)
    return params


def get_labels(params):
    """
    return the labels for every parameter in LaTex code.

    Parameters
    ----------
    params: list of string
        list with parameters names (possible prefixes included)

    Returns
    -------
    labels: dict
        dictionary with parameter names as keys and LaTex code as values.
    """
    all_labels = {
        "c0": r"$C_0$",
        "Cs": r"$C_\mathrm{stray}$",
        "em": r"$\varepsilon_\mathrm{m}$",
        "km": r"$\sigma_\mathrm{m}$",
        "sigma": r"$\sigma$",
        "kcp": r"$\sigma_\mathrm{cp}$",
        "ecp": r"$\varepsilon_\mathrm{cp}$",
        "kmed": r"$\sigma_\mathrm{med}$",
        "emed": r"$\varepsilon_\mathrm{med}$",
        "p": r"$p$",
        "dm": r"$d_\mathrm{m}$",
        "Rc": r"$R_\mathrm{c}$",
        "Rk": r"$R_\mathrm{k}$",
        "ene": r"$\varepsilon_\mathrm{ne}$",
        "kne": r"$\sigma_\mathrm{ne}$",
        "knp": r"$\sigma_\mathrm{np}$",
        "enp": r"$\varepsilon_\mathrm{np}$",
        "dn": r"$d_\mathrm{n}$",
        "Rn": r"$R_\mathrm{n}$",
        "k": r"$\kappa$",
        "epsl": r"$\varepsilon_\mathrm{l}$",
        "tau": r"$\tau$",
        "tauE": r"$\tau_\mathrm{E}$",
        "tauk": r"$\tau_\mathrm{k}$",
        "a": r"$a$",
        "alpha": r"$\alpha$",
        "kdc": r"$\sigma_\mathrm{DC}$",
        "__lnsigma": r"$\ln\sigma$",
        "L": r"$L$",
        "C": r"$C$",
        "Cd": r"$C_\mathrm{d}$",
        "R": r"$R$",
        "Rd": r"$R_\mathrm{d}$",
        "Rinf": r"$R_\infty$",
        "epsinf": r"$\varepsilon_\infty$",
        "R0": r"$R_0$",
        "RE": r"$R_\mathrm{E}$",
        "eps": r"$\varepsilon_\mathrm{r}$",
    }

    # for 4 cole cole model
    for i in range(1, 5):
        all_labels["tau" + str(i)] = rf"$\tau_{i}$"
        all_labels["deps" + str(i)] = rf"$\Delta\varepsilon_{i}$"
        all_labels["a" + str(i)] = rf"$a_{i}$"

    labels = {}
    for p in params:
        if p != "__lnsigma":
            tmp = p.split("_")
            pre = ""
        else:
            tmp = ["__lnsigma"]
        if len(tmp) == 2:
            pre = tmp[0]
            par = tmp[1]
        elif len(tmp) == 1:
            par = tmp[0]
        else:
            raise RuntimeError(
                f"The parameter {p} cannot be split in prefix and suffix."
            )
        try:
            label = all_labels[par]
        except KeyError:
            print(f"No LaTex representation for parameter {par} implemented.")
        labels[p] = rf"{pre} {label}"
    return labels


def available_models():
    """Return list of available models."""
    models = [
        "ColeCole",
        "ColeColeR",
        "ColeCole4",
        "ColeCole3",
        "ColeCole2",
        "ECISLoFerrier",
        "HavriliakNegami",
        "Raicu",
        "Randles",
        "RandlesCPE",
        "RC",
        "Debye",
        "ParticleSuspension",
        "ParticleSuspensionBH",
        "SingleShell",
        "SingleShellEllipsoid",
        "SingleShellBH",
        "SingleShellWall",
        "SingleShellWallEllipsoid",
        "SingleShellWallBH",
        "DoubleShell",
        "DoubleShellEllipsoid",
        "DoubleShellBH",
        "DoubleShellWall",
        "DoubleShellWallEllipsoid",
        "DoubleShellWallBH",
        "CPE",
        "CPEonset",
        "CPECT",
        "CPECTW",
        "DRC",
        "L",
        "R",
        "C",
        "W",
        "Wo",
        "ADIbR",
        "ADIaR",
        "ADIIR",
        "ADIbA",
        "ADIaA",
        "ADIIA",
        "LCR",
        "LR",
        "LRSkin",
        "Ws",
        "Cstray",
    ]
    return models


def available_file_format():
    """List available file formats.

    Currently available:

    **XLSX** and **CSV**:

    The file is structured like:
    frequency, real part of impedance, imaginary part of impedance.
    There may be many different sets of impedance data,
    i.e. there may be more columns with the real and the imaginary part.
    Then, the frequencies column must not be repeated.
    In fact, the number of columns equals the number of
    impedance data sets plus one (for the frequency).

    .. note::

        A single header line is needed in a CSV and XLSX file.
        It may contain for example `frequency, Real Part, Imag Part`.
        Otherwise the read-in function will fail.

    **CSV_E4980AL**:

    Read in data that is structured in 5 columns:
    frequency, real part, imaginary part of the impedance, voltage, current

    .. note::

        There is always only one data set in a file.

    **TXT**:

    These files contain frequency, real and imaginary part of the impedance
    (i.e., 3 columns).
    The TXT files may contain two traces; only one of them is read in.
    For TXT files you can specify the number of rows to skip.
    Moreover, the file ending is not strictly enforced here.

    **DF**:

    The dataframe is structured like:
    frequency, real part of impedance, imaginary part of impedance.
    This is not a file format, so the workflow is different.
    The 'directory' parameter must be set to None and the following
    parameter must be added to `**kwargs`: 'df', 'df_freq_column',
    'df_real_column' and 'df_imag_column'. The 'df' parameter
    is the dataframe object, and the other parameters are the
    column names of the dataframe to be read in.

    .. note::

        The dataframe can contain multiple impedance data sets.
        While the dataframe object and the frequency, real and imaginary
        columns are not None, the dataframe will be used instead of a file.

    See Also
    --------
    :class:`impedancefitter.fitter.Fitter`
    """
    formats = ["XLSX", "CSV", "CSV_E4980AL", "TXT", "DF"]
    return formats


def _model_function(modelname):
    """Wrapper to return correct function for model."""
    model_name_to_function = {
        "ColeCole": cole_cole_model,
        "ColeCole4": cole_cole_4_model,
        "ColeCole3": cole_cole_3_model,
        "ColeCole2": cole_cole_2_model,
        "ColeCole2Tissue": cole_cole_2tissue_model,
        "HavriliakNegami": havriliak_negami,
        "HavriliakNegamiTissue": havriliak_negamitissue,
        "Raicu": raicu,
        "ColeColeR": cole_cole_R_model,
        "ECISLoFerrier": Z_ECIS_Lo_Ferrier,
        "Randles": Z_randles,
        "RandlesCPE": Z_randles_CPE,
        "DRC": drc_model,
        "RC": RC_model,
        "LossyDielectric": rc_model,
        "RCtau": rc_tau_model,
        "ParticleSuspension": particle_model,
        "ParticleSuspensionBH": particle_bh_model,
        "SingleShell": single_shell_model,
        "SingleShellEllipsoid": single_shell_ellipsoid_model,
        "SingleShellWallEllipsoid": single_shell_wall_ellipsoid_model,
        "SingleShellWall": single_shell_wall_model,
        "SingleShellWallBH": single_shell_wall_bh_model,
        "SingleShellBH": single_shell_bh_model,
        "DoubleShell": double_shell_model,
        "DoubleShellEllipsoid": double_shell_ellipsoid_model,
        "DoubleShellWall": double_shell_wall_model,
        "DoubleShellWallEllipsoid": double_shell_wall_ellipsoid_model,
        "DoubleShellWallBH": double_shell_wall_bh_model,
        "DoubleShellBH": double_shell_bh_model,
        "CPE": cpe_model,
        "CPETissue": cpetissue_model,
        "CPEonset": cpe_onset_model,
        "CPECTTissue": cpe_ct_tissue_model,
        "CPECT": cpe_ct_model,
        "CPECTW": cpe_ct_w_model,
        "R": Z_R,
        "C": Z_C,
        "L": Z_L,
        "W": Z_w,
        "Wo": Z_wo,
        "ADIbR": Z_ADIb_r,
        "ADIaR": Z_ADIa_r,
        "ADIIR": Z_ADII_r,
        "ADIbA": Z_ADIb_a,
        "ADIaA": Z_ADIa_a,
        "ADIIA": Z_ADII_a,
        "Ws": Z_ws,
        "LCR": Z_loss,
        "LR": Z_in,
        "LRSkin": Z_skin,
        "Cstray": Z_stray,
    }
    if modelname not in model_name_to_function:
        raise NotImplementedError(f"Model {modelname} not implemented")
    return model_name_to_function[modelname]


def _process_parallel(model):
    """Process parallel circuit.

    Parameters
    ----------
    model: list
        Contains the parallel circuit as a nested list.

    Returns
    -------
    :py:class:`lmfit.model.CompositeModel`
        The CompositeModel of the parallel circuit.
    """
    assert len(model) == 3, "The model must be [model1, ',' , model2]"
    first_model = model[0]
    second_model = model[2]
    first = _process_element(first_model)
    second = _process_element(second_model)
    return CompositeModel(first, second, parallel)


def _generate_model(m):
    """Generate a :py:class:`lmfit.model.Model` from a Python function."""
    if "_" in m:
        # get possible prefix
        info = m.split("_")
        if len(info) != 2:
            raise RuntimeError(
                f"A model must be always named PREFIX_MODEL! Instead you named it: {m}"
            )
        # also checks if model exists
        return Model(_model_function(info[0].strip()), prefix=info[1].strip() + "_")
    else:
        return Model(_model_function(m.strip()))


def _process_element(c):
    """Process an individual element of the circuit.

    Parameters
    ----------
    c: str or list
        either a model is generated or the circuit is further processed
    """
    if isinstance(c, str):
        return _generate_model(c)
    elif isinstance(c, list):
        return _process_circuit(c)
    else:
        raise RuntimeError


def _process_series(circuitstr):
    """Process series circuit.

    Parameters
    ----------
    circuitstr: list, str
        string representation of circuit, must be a list

    Returns
    -------
    :py:class:`lmfit.model.CompositeModel`
        the composite model representation of the series
    """
    circuit = []
    for c in circuitstr:
        if c == "+":
            continue
        element = _process_element(c)
        circuit.append(element)
    composite_model = circuit[0]
    circuit.pop(0)
    for m in circuit:
        composite_model += m

    return composite_model


def _process_circuit(circuit):
    """Generate LMFIT model from circuit.

    Parameters
    ----------
    circuit: list
        Nested list representation of circuit.

    Returns
    -------
    :py:class:`lmfit.model.CompositeModel` or :class:`lmfit.model.Model`
        the final model of the entire circuit
    """
    logger.debug(f"circuit: {circuit}")
    if isinstance(circuit, str):
        circuit = [circuit]
    if not isinstance(circuit, list):
        raise RuntimeError("You must have entered a wrong circuit!")

    # if there are elements in series or only one element
    if "+" in circuit or len(circuit) == 1:
        c = _process_series(circuit)
    elif "," in circuit:
        c = _process_parallel(circuit)
    else:
        raise RuntimeError("You must have entered a wrong circuit!")
    return c


def _check_circuit(circuit, startpar=False):
    if circuit.count("(") != circuit.count(")"):
        raise RuntimeError(
            """You must have entered a wrong circuit!
                              There are parentheses that do not match!
                              For example, `parallel(R, C`"""
        )
    # check if circuit contains just one element
    if isinstance(circuit, str):
        match = bool(re.match(r"[a-zA-Z_0-9]", circuit))
        if not match:
            raise RuntimeError("You must have entered a wrong circuit!")
        else:
            return
    # check for case with pure parallel or series
    if all(isinstance(c, str) for c in circuit):
        # check that only one type exists then
        if "+" in circuit:
            if "," in circuit:
                raise RuntimeError("You must have entered a wrong circuit!")
            # if it started with parallel, raise an error.
            # here we catch `parallel(R + C)`, for example
            if startpar:
                raise RuntimeError("You must have entered a wrong circuit!")
        else:
            if "," not in circuit:
                raise RuntimeError("You must have entered a wrong circuit!")

    for c in circuit:
        if isinstance(c, str):
            match = bool(re.match(r"[a-zA-Z_0-9]", c))
            if match or c == "," or c == "+":
                continue
            else:
                raise RuntimeError("You must have entered a wrong circuit!")
        elif isinstance(c, list):
            # we only have a list if there is any parallel circuit in the circuit
            if circuit.count(",") != 1:
                # if we start with a parallel element,
                # the first entry in the list must contain a comma
                if startpar and circuit[0].count(",") != 1:
                    raise RuntimeError(
                        "You must have entered a wrong circuit! "
                        "There is a comma missing."
                    )
                # otherwise there must be one comma somewhere in the circuit
                if c.count(",") != 1:
                    raise RuntimeError(
                        "You must have entered a wrong circuit! "
                        "There is a comma missing."
                    )
            _check_circuit(c)


def get_equivalent_circuit_model(modelname, logscale=False, diel=False):
    """Get LMFIT CompositeModel.

    Parameters
    ----------
    modelname: str
        String representation of the equivalent circuit.
    logscale: bool
        Convert to logscale.
    diel: bool
        Convert to complex permittivity and fit this instead of impedance.

    Returns
    -------
    :py:class:`lmfit.model.CompositeModel` or :class:`lmfit.model.Model`
        the final model of the entire circuit

    Notes
    -----
    The parser is based on Pyparsing.
    It is sensitive towards extra `(` or `)` or `+`.
    Thus, keep the circuit simple.
    """
    circuit = []
    assert isinstance(modelname, str), "Pass the model as a string"
    str2parse = modelname.replace("parallel", "")
    circuit_elements = pp.Word(pp.srange("[a-zA-Z_0-9]"))
    plusop = pp.Literal("+")
    commaop = pp.Literal(",")
    expr = pp.infixNotation(
        circuit_elements, [(plusop, 2, pp.opAssoc.LEFT), (commaop, 2, pp.opAssoc.LEFT)]
    )
    try:
        circuitstr = expr.parseString(str2parse)
    except pp.ParseException:
        raise ("You must provide a correct string!")
    _check_circuit(circuitstr.asList()[0], startpar=modelname.startswith("parallel"))
    circuit = _process_circuit(circuitstr.asList()[0])
    if logscale:
        circuit = CompositeModel(circuit, Model(dummy), log)
    elif diel:
        circuit = CompositeModel(circuit, Model(make_eps), eps)
    if logscale and diel:
        raise RuntimeError("You must chose the representation of the impedance value")
    _check_models_suffix(circuit)
    logger.debug(f"Created composite model {circuit}")
    return circuit


def _check_models_suffix(circuit):
    baseparams = []
    suffixes = {}
    for p in circuit.param_names:
        tmp = p.split("_")
        suf = None
        if len(tmp) == 2:
            suf = tmp[0]
            par = tmp[1]
        elif len(tmp) == 1:
            par = tmp[0]
        else:
            raise RuntimeError(
                f"The parameter {p} cannot be split in prefix and suffix."
            )
        # fill baseparams list and check suffixes
        if par in baseparams:
            if len(suffixes[par]) == 0 or suf is None:
                raise RuntimeError(
                    f"There is an error with the parameter {p}. "
                    "Please make sure that all parameters of "
                    "this type have a proper prefix according to "
                    "its model's suffix. "
                    "Otherwise, wrong parameter assignments will happen."
                )
        else:
            baseparams.append(par)
            suffixes[par] = []
            if suf is not None:
                suffixes[par].append(suf)


def dummy(omega):
    """Dummy function returning only ones."""
    return np.ones(omega.shape)


def make_eps(omega, c0all):
    """Compute permittivity."""
    return 1.0 / (1j * omega * c0all)


def _model_label(model):
    """Return label strings for each model."""
    labels = {
        "ColeCole": "Cole-Cole",
        "ColeColeR": "Cole-Cole w/ R",
        "ColeCole4": "4 Cole-Cole",
        "ColeCole3": "3 Cole-Cole",
        "ColeCole2": "2 Cole-Cole",
        "HavriliakNegami": "Havriliak-Negami",
        "Raicu": "Raicu",
        "Randles": "Randles",
        "RandlesCPE": "Randles w/ CPE",
        "RC": "RC",
        "RCtau": "RCtau",
        "LossyDielectric": "Lossy dielectric",
        "ParticleSuspension": "Particle suspension",
        "ParticleSuspensionBH": "Particle suspension Bruggeman Hanai",
        "SingleShell": "Single Shell",
        "SingleShellBH": "Single Shell Bruggeman Hanai",
        "SingleShellWall": "Single Shell Wall",
        "SingleShellWallBH": "Single Shell Wall Bruggeman Hanai",
        "DoubleShell": "Double Shell",
        "DoubleShellBH": "Double Shell Bruggeman Hanai",
        "DoubleShellWall": "Double Shell Wall",
        "DoubleShellWallBH": "Double Shell Wall Bruggeman Hanai",
        "CPE": "CPE",
        "CPETissue": "CPE for tissues",
        "CPEonset": "CPEonset",
        "CPECT": "CPECT",
        "CPECTTissue": "CPECT for tissues",
        "CPECTW": "CPECTW",
        "DRC": "DRC",
        "L": "L",
        "R": "R",
        "C": "C",
        "Cstray": "C [pF]",
        "W": "W",
        "Wo": "W open",
        "Ws": "W short",
        "LCR": "LCR",
        "LR": "LR",
    }

    try:
        label = labels[model]
    except KeyError:
        print(f"There has not yet been a label defined for the model {model}")
    return label


def _get_element(name):
    resistors = ["R", "Rk"]
    capacitors = ["C", "Cstray"]
    resistorlike = [
        "ColeCole",
        "ColeColeR",
        "ColeCole4",
        "ColeCole3",
        "ColeCole2",
        "Raicu",
        "HavriliakNegami",
        "Randles",
        "RandlesCPE",
        "RC",
        "RCtau",
        "LossyDielectric",
        "DRC",
        "ParticleSuspension",
        "ParticleSuspensionBH",
        "SingleShell",
        "SingleShellBH",
        "SingleShellWall",
        "SingleShellWallBH",
        "DoubleShell",
        "DoubleShellBH",
        "DoubleShellWall",
        "DoubleShellWallBH",
    ]
    capacitorlike = [
        "CPE",
        "CPEonset",
        "CPETissue",
        "CPECTTissue",
        "CPECT",
        "CPECTW",
        "W",
        "Wo",
        "Ws",
    ]
    inductors = ["L"]
    inductorlike = ["LR", "LCR"]

    tmp = name.split("_")
    par = tmp[0]
    label = _model_label(par)
    if len(tmp) == 2:
        pre = tmp[1]
        label += "_" + pre

    if par in resistors:
        element = elm.ResistorIEC
    elif par in resistorlike:
        element = elm.ResistorIEEE
    elif par in capacitors:
        element = elm.Capacitor
    elif par in capacitorlike:
        element = elm.Capacitor2
    elif par in inductors:
        element = elm.Inductor
    elif par in inductorlike:
        element = elm.Inductor2
    return element, label


def draw_scheme(modelname, show=True, save=False):
    """Show (and save) SchemDraw drawing.

    Parameters
    ----------
    modelname: str
        String representation of the equivalent circuit.
    show: bool, optional
        Show scheme in matplotlib window.
    save: bool, optional
        Save scheme to file. File is called `scheme.svg`.
    """
    # read and check circuit
    assert isinstance(modelname, str), "Pass the model as a string"
    str2parse = modelname.replace("parallel", "")
    circuit_elements = pp.Word(pp.srange("[a-zA-Z_0-9]"))
    plusop = pp.Literal("+")
    commaop = pp.Literal(",")
    expr = pp.infixNotation(
        circuit_elements, [(plusop, 2, pp.opAssoc.LEFT), (commaop, 2, pp.opAssoc.LEFT)]
    )
    try:
        circuitstr = expr.parseString(str2parse)
    except pp.ParseException:
        raise ("You must provide a correct string!")
    _check_circuit(circuitstr.asList()[0], startpar=modelname.startswith("parallel"))

    # start drawing
    # TODO check
    d = schemdraw.Drawing()
    """
    d.set_anchor("start")
    d, endpts = _cycle_circuit(
        circuitstr.asList()[0], d, endpts=(source.start, source.end), depth=0
    )
    # finalize drawing
    endpts = (Point((endpts[0][0], 0.0)), Point((endpts[1][0], 0.0)))
    d.add(elm.SourceSin, d="left", label="Impedance analyzer", tox=start)
    """
    d.draw(show=show)
    if save:
        d.save("scheme.svg")


def _cycle_circuit(circuit, d, endpts, step=3.0, depth=0):
    """Generate sketch for (sub-)circuit.

    Parameters
    ----------
    circuit: list
        Nested list representation of circuit.
    d: Drawing
        SchemDraw Drawing object.
    endpts: tuple
        Coordinates where to start and end
    step: float
        Step width
    depth: int
        Level of depth

    Returns
    -------
    Drawing:
        the final drawing of the entire (sub-)circuit

    Notes
    -----
    The following circuits were tested with this code:

    .. code-block::

        import impedancefitter as ifit

        models = ["RC + R + parallel(R, C) + parallel(R, C)",
                  "parallel(R + C, parallel(R, C))",
                  "RC + R + parallel(R + C, C + R) + parallel(R, C)",
                  "RC + R + parallel(R, parallel(R, CPE))",
                  "RC + R + parallel(parallel(R,C), parallel(R, C))",
                  "RC + R + parallel(R + C, parallel(R, C))",
                  "RC + R + parallel(R + C, parallel(R, C))",
                  "parallel(RC_f1 + parallel(R_f3, C_f4),  Cstray)"]
        for m in models:
            ifit.utils.draw_scheme(m)

    Please file an issue if one of your circuits is not drawn properly.
    """
    logger.debug(f"circuit: {circuit}")
    if isinstance(circuit, str):
        circuit = [circuit]
    if not isinstance(circuit, list):
        raise RuntimeError("You must have entered a wrong circuit!")

    # if there are elements in series or only one element
    if "+" in circuit or len(circuit) == 1:
        d, endpts = _add_series_drawing(circuit, d, endpts, step=step, depth=depth)
    elif "," in circuit:
        d, endpts = _add_parallel_drawing(circuit, d, endpts, depth=depth)
    else:
        raise RuntimeError("You must have entered a wrong circuit!")
    return d, endpts


def _draw_element(c, d, endpts, step=3.0, depth=0):
    if isinstance(c, str):
        element, label = _get_element(c)
        endpts = (endpts[0], [endpts[0][0] + step, endpts[1][1]])
        e = d.add(element, d="right", label=label, endpts=endpts)
        # increment position
        endpts = (e.end, e.end)
    elif isinstance(c, list):
        d, endpts = _cycle_circuit(c, d, endpts, step, depth=depth + 1)
    return d, endpts


def _add_series_drawing(circuit, d, endpts, step=3.0, depth=0):
    for c in circuit:
        if c == "+":
            continue
        if depth == 0:
            endpts = (Point((endpts[0][0], 0.0)), Point((endpts[1][0], 0.0)))

        d, endpts = _draw_element(c, d, endpts, step, depth=depth)
    return d, endpts


def _add_parallel_drawing(circuit, d, endpts, depth=0):
    assert len(circuit) == 3, "The model must be [model1, ',' , model2]"

    first_element = circuit[0]
    second_element = circuit[2]

    # add first element
    anchorx1 = deepcopy(endpts[0][0])
    anchory1 = deepcopy(endpts[0][1])

    # default
    step = 3.0
    # account for possible overlength of previous element
    distance = endpts[1][0] - endpts[0][0]
    if distance > 0:
        step = _get_step_width(first_element, distance)

    d, endpts = _draw_element(first_element, d, endpts, step=step, depth=depth)

    anchorx2 = deepcopy(endpts[0][0])
    anchory2 = deepcopy(endpts[1][1])

    distance = anchorx2 - anchorx1
    assert distance > 0, "There is something wrong with your circuit."
    # find longest series in 2nd element
    longest = _determine_longest_length(second_element)
    if not np.less_equal(longest * 3.0, distance):
        raise RuntimeError(
            """There is a subcircuit in a parallel circuit
                              that is longer than the first circuit.
                              For example: `parallel(R_f1, C + R)`.
                              If you want to draw such a circus, reformulate it
                              as `parallel(C + R, R_f1)`"""
        )

    step = _get_step_width(second_element, distance)
    # add second element
    endpts = ([anchorx1, anchory1], [anchorx1, anchory2 - 3.0])

    d.add(elm.LINE, d="down", endpts=endpts)

    endpts = ([anchorx1, anchory2 - 3.0], [anchorx2, anchory2 - 3.0])

    d, endpts = _draw_element(second_element, d, endpts=endpts, step=step, depth=depth)

    anchory2 = deepcopy(endpts[1][1])
    endpts = ([endpts[1][0], endpts[1][1]], [endpts[1][0], anchory1])
    d.add(elm.LINE, d="up", endpts=endpts)
    endpts = ([anchorx2, anchory2], [anchorx2, anchory2])

    return d, endpts


def _determine_longest_length(circuit):
    longest_length = 0
    counter = [0, 0]
    side = 0
    if isinstance(circuit, str):
        return 1
    for c in circuit:
        if isinstance(c, str):
            if c != "+" and c != ",":
                counter[side] += 1
            elif c == ",":
                side += 1
        elif isinstance(c, list):
            tmp = _determine_longest_length(circuit)
            if tmp > longest_length:
                longest_length = tmp

    if max(counter) > longest_length:
        return max(counter)
    else:
        return longest_length


def _get_step_width(circuit, distance):
    counter = 0
    if isinstance(circuit, str):
        return distance
    for c in circuit:
        if c == ",":
            break
        elif c != "+":
            counter += 1
    step = distance / counter
    return step


def _return_resistance_capacitance(omega, Z):
    R = Z.real
    C = -1.0 / (omega * Z.imag)
    return R, C


def KK_integral_transform(omega, Z):
    """Kramers-Kronig integral transform.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`, double
        frequency array
    Z: :class:`numpy.ndarray`, complex
        impedance array

    Returns
    -------
    :class:`numpy.ndarray`, complex
        The transformed impedance array.

    Notes
    -----
    Implementation following [Urquidi1990]_

    .. [Urquidi1990] Urquidi-Macdonald, M., Real, S., & Macdonald, D. D. (1990).
                     Applications of Kramers-Kronig transforms in the analysis
                     of electrochemical impedance data-III. Stability and linearity.
                     Electrochimica Acta, 35(10), 1559–1566.
                     https://doi.org/10.1016/0013-4686(90)80010-L
    """
    ZKK = np.ndarray(omega.shape, dtype=complex)
    for i, w in enumerate(omega):
        x = np.append(omega[:i], omega[i + 1 :])
        real = np.append(Z.real[:i], Z.real[i + 1 :])
        imag = np.append(Z.imag[:i], Z.imag[i + 1 :])
        # real part
        integrand = (x * imag - w * Z.imag[i]) / (x * x - w * w)
        ZKK[i] = -2.0 / np.pi * simpson(integrand, x=x)
        # imag part
        integrand = (real - Z.real[i]) / (x * x - w * w)
        ZKK[i] += 1j * 2.0 * w / np.pi * simpson(integrand, x=x)
    return ZKK


def save_impedance(omega, impedance, format="CSV", filename="impedance"):
    """Save impedance to CSV or XLSX file.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`, double
        frequency array
    impedance: :class:`numpy.ndarray`, complex
        impedance array
    format: str
        use either CSV or XLSX format.
        Based on the format, the correct ending is chosen.
    filename: str
        specify a filename (without ending!).
        The default is impedance.csv or impedance.xlsx
    """
    assert isinstance(filename, str), "You need to provide a str as filename!"
    outdict = {
        "freq": omega / (2.0 * np.pi),
        "real": impedance.real,
        "imag": impedance.imag,
    }
    df = pd.DataFrame(outdict)
    if format == "CSV":
        df.to_csv(filename + ".csv", index=False)
    elif format == "XLSX":
        df.to_excel(filename + ".xlsx", index=False)
    else:
        raise (RuntimeError("You provided a wrong format. Use either CSV or XLSX."))
