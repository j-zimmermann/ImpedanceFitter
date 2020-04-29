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


import numpy as np
import yaml
import logging
from scipy.constants import epsilon_0 as e0
from collections import Counter
from .elements import Z_C, Z_stray, Z_in, Z_loss, parallel, Z_R, Z_L, Z_w, Z_ws, Z_wo
from .cole_cole import cole_cole_model
from .cole_cole_R import cole_cole_R_model
from .double_shell import double_shell_model
from .randles import Z_randles, Z_randles_CPE
from .rc import rc_model
from .RC import RC_model
from .cpe import cpe_model, cpe_ct_model, cpe_ct_w_model
from .single_shell import single_shell_model
from .drc import drc_model
from lmfit import Model, CompositeModel

logger = logging.getLogger('impedancefitter-logger')


def return_diel_properties(omega, Z, c0):
    r"""
    return relative permittivity and conductivity from impedance spectrum
    Use that the impedance is

    .. math::

        Z = (j \omega \varepsilon^\ast)^{-1} ,

    where :math:`\varepsilon^\ast` is the complex permittivity (see for instance the paper with DOI 10.1063/1.1722949
    for further explanation).

    The relative permittivity is the real part of :math:`\varepsilon^\ast` divided by the vacuum permittivity and
    the conductivity is the imaginary part times the frequency.

    Parameters
    ----------

    omega: double or ndarray of double
        frequency array
    Z: complex or array of complex
        impedance array
    c0: double
        unit capacitance of device

    Returns
    -------

    eps_r: double
        relative permittivity
    conductivity: double
        conductivity in S/m
    """
    epsc = 1. / (1j * omega * Z * c0)
    eps_r = epsc.real / e0
    conductivity = epsc.imag * omega
    return eps_r, conductivity


def check_parameters(bufdict):
    """
    check parameters for physical correctness

    Parameters
    ---------

    bufdict: dict
        Contains all parameters and their values

    .. todo::
        this needs to work prefixes
    """
    # capacitances in pF
    capacitances = ['c0', 'C_stray']
    for c in capacitances:
        try:
            assert not np.isclose(bufdict[c].value, 0.0, atol=1e-5), """{} is used in pF, do you really want it to be that small?
                                                                            It might be ignored in the analysis""".format(c)
        except KeyError:
            pass

    # check volume fraction
    try:
        assert 0 <= bufdict['p'].value <= 1.0, "p is the volume fraction and needs to be between 0.0 and 1.0. Change your initial value."
        if bufdict['p'].vary:
            assert 0 <= bufdict['p'].min <= 1.0, "p is the volume fraction and needs to be between 0.0 and 1.0. Change the min value accordingly."
            assert 0 <= bufdict['p'].max <= 1.0, "p is the volume fraction and needs to be between 0.0 and 1.0. Change the max value accordingly."
    except KeyError:
        pass

    # check tau (used in ColeCole models
    try:
        assert not np.isclose(bufdict['tau'].value, 0.0, atol=1e-5), "tau is used in ps, do you really want it to be that small?"
    except KeyError:
        pass

    # check permittivities
    permittivities = ['em', 'ecp', 'emed', 'ene', 'enp', 'el', 'eh', 'eps']
    for p in permittivities:
        try:
            assert bufdict[p].value >= 1.0, "The permittivity {} needs to be greater than or equal to 1. Change the initial value.".format(p)
            if bufdict[p].vary:
                assert bufdict[p].min >= 1.0, "The permittivity {} needs to be greater than or equal to 1. Change the min value.".format(p)
                assert bufdict[p].max >= 1.0, "The permittivity {} needs to be greater than or equal to 1. Change the max value.".format(p)
        except KeyError:
            pass

    # check special parameters
    exponents = ['a', 'alpha']
    for e in exponents:
        try:
            assert 0 <= bufdict[e].value <= 1.0, "{} is an exponent that needs to be between 0.0 and 1.0. Change your initial value.".format(e)
            if bufdict[p].vary:
                assert 0 <= bufdict[e].min >= 0.0, "{} is an exponent that needs to be between 0.0 and 1.0. Change your min value.".format(e)
                assert 0 <= bufdict[e].max <= 1.0, "{} is an exponent that needs to be between 0.0 and 1.0. Change your max value.".format(e)
        except KeyError:
            pass

    for p in bufdict:
        # __lnsigma can be negative
        if p == '__lnsigma':
            continue
        assert bufdict[p].value >= 0.0, "{} needs to be positive. Change your initial value.".format(p)
        if bufdict[p].vary:
            if bufdict[p].min <= 0.0:
                logger.debug("{} needs to be positive. Changed your min value to 0.".format(p))
                bufdict[p].set(min=0.0)
            assert bufdict[p].max >= 0.0, "{} needs to be positive. Change your max value.".format(p)

    return bufdict


def set_parameters(model, parameterdict=None, emcee=False):
    """
    Parameters
    -----------

    model: string
        LMFIT model object.
    parameterdict: optional
        a dictionary containing parameters for model with *min*, *max*, *vary* info for LMFIT.
        If it is None (default), the parameters are read in from a yaml-file.
    emcee: bool, optional
        if emcee is used, an additional `__lnsigma` parameter will be set


    Returns:
    --------

    params: Parameters
        LMFIT `Parameters`.
    """
    if parameterdict is None:
        try:
            infile = open('input.yaml', 'r')
            bufdict = yaml.safe_load(infile)
        except OSError:
            logger.error("Please provide a yaml-input file.")
            raise
    else:
        try:
            bufdict = parameterdict.copy()
        except KeyError:
            logger.error("Your parameterdict lacks an entry for the model. Required are: {}".format(model.param_names))
            raise

    bufdict = _clean_parameters(bufdict, model.param_names)
    logger.debug("Setting values for parameters {}".format(model.param_names))
    logger.debug("Parameters: {}".format(bufdict))
    for key in model.param_names:
        model.set_param_hint(key, **bufdict[key])
    parameters = model.make_params()
    parameters = check_parameters(parameters)
    if emcee and '__lnsigma' not in parameterdict:
        logger.warning("""You did not provide the parameter '__lnsigma'.
                          It is needed for the emcee of unweighted data (as implemented here).
                          We now use the lmfit default:\n
                          value=np.log(0.1), min=np.log(0.001), max=np.log(2)""")
        parameters.add("__lnsigma", value=np.log(0.1), min=np.log(0.001), max=np.log(2.0))
    elif emcee and "__lnsigma" in parameterdict:
        parameters.add("__lnsigma", **parameterdict["__lnsigma"])

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

    assert Counter(names) == Counter(params.keys()), "You need to provide the following parameters (maybe with prefixes)" + str(names)
    return params


def get_labels(params):
    """
    return the labels for every parameter in LaTex code.

    Parameters
    ----------

    params: list of string
        list with parameters names (possible prefixes included
    Returns
    -------

    labels: dict
        dictionary with parameter names as keys and LaTex code as values.
    """
    all_labels = {
        'c0': r'$C_0$',
        'C_stray': r'$C_\mathrm{stray}$',
        'em': r'$\varepsilon_\mathrm{m}$',
        'km': r'$\sigma_\mathrm{m}$',
        'kcp': r'$\sigma_\mathrm{cp}$',
        'ecp': r'$\varepsilon_\mathrm{cp}$',
        'kmed': r'$\sigma_\mathrm{med}$',
        'emed': r'$\varepsilon_\mathrm{med}$',
        'p': r'$p$',
        'dm': r'$d_\mathrm{m}$',
        'Rc': r'$R_\mathrm{c}$',
        'ene': r'$\varepsilon_\mathrm{ne}$',
        'kne': r'$\sigma_\mathrm{ne}$',
        'knp': r'$\sigma_\mathrm{np}$',
        'enp': r'$\varepsilon_\mathrm{np}$',
        'dn': r'$d_\mathrm{n}$',
        'Rn': r'$R_\mathrm{n}$',
        'k': r'$\kappa$',
        'el': r'$\varepsilon_\mathrm{l}$',
        'tau': r'$\tau$',
        'tauE': r'$\tau_\mathrm{E}$',
        'a': r'$a$',
        'alpha': r'$\alpha$',
        'kdc': r'$\sigma_\mathrm{DC}$',
        'eh': r'$\varepsilon_\mathrm{h}$',
        '__lnsigma': r'$\ln\sigma$',
        'L': r'$L$',
        'C': r'$C$',
        'Cd': r'$C_\mathrm{d}$',
        'R': r'$R$',
        'Rd': r'$R_\mathrm{d}$',
        'Rinf': r'$R_\infty$',
        'R0': r'$R_0$',
        'RE': r'$R_\mathrm{E}$',
        'eps': r'$\varepsilon_\mathrm{r}$'}

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
            raise RuntimeError("The parameter {} cannot be split in prefix and suffix.".format(p))
        try:
            label = all_labels[par]
        except KeyError:
            print("There has not yet been a LaTex representation of the parameter {} implemented.".format(par))
        labels[p] = r"{} {}".format(pre, label)
    return labels


def available_models():
    """
    return list of available models
    """
    models = ['ColeCole',
              'ColeColeR',
              'Randles',
              'RandlesCPE',
              'RCfull',
              'RC',
              'SingleShell',
              'DoubleShell',
              'CPE',
              'CPECT',
              'CPECTW',
              'DRC',
              'L',
              'R',
              'C',
              'W',
              'Wo',
              'Ws']
    return models


def model_information(model):
    """Get detailed information about model.

    Parameters
    -----------
    model: string
        name of model, see available models from :func:`available_models`.

    Returns:
    --------
    description: string
        Description what is behind the model, where it can be found in literature and where it is implemented.

    .. todo::
        Update documentation here.

    """
    information = {'ColeCole': """ Cole-Cole model as implemented in paper with DOI:10.1063/1.4737121.
                                    You need to provide the unit capacitance of your device to get the dielectric properties of
                                    the Cole-Cole model. """,
                   'ColeColeR': """ Standard Cole-Cole circuit as given in paper with DOI:10.1016/b978-1-4832-3111-2.50008-0.
                                      Implemented in :func:impedancefitter.cole_cole_R.co """,
                   'Randles': """
                                """,
                   'RandlesCPE': """
                                """,
                   'RCfull': """
                   """,
                   'RC': """
                   """,
                   'SingleShell': """
                   """,
                   'DoubleShell': """
                   """}

    if model in information:
        description = information['model']
    else:
        raise NotImplementedError("This model has not been implemented yet or no information is available.")
    return description


def available_file_format():
    """
    .. todo::
        Update documentation here.

    """
    formats = ['XLSX', 'CSV', 'CSV_E4980AL', "TXT"]
    return formats


def model_function(modelname):
    if modelname == 'ColeCole':
        model = cole_cole_model
    elif modelname == 'ColeColeR':
        model = cole_cole_R_model
    elif modelname == 'Randles':
        model = Z_randles
    elif modelname == 'RandlesCPE':
        model = Z_randles_CPE
    elif modelname == 'DRC':
        model = drc_model
    elif modelname == 'RCfull':
        model = RC_model
    elif modelname == 'RC':
        model = rc_model
    elif modelname == 'SingleShell':
        model = single_shell_model
    elif modelname == 'DoubleShell':
        model = double_shell_model
    elif modelname == 'CPE':
        model = cpe_model
    elif modelname == 'CPECT':
        model = cpe_ct_model
    elif modelname == 'CPECTW':
        model = cpe_ct_w_model
    elif modelname == "R":
        model = Z_R
    elif modelname == "C":
        model = Z_C
    elif modelname == "L":
        model = Z_L
    elif modelname == "W":
        model = Z_w
    elif modelname == "Wo":
        model = Z_wo
    elif modelname == "Ws":
        model = Z_ws
    elif modelname == "LCR":
        model = Z_loss
    elif modelname == "LR":
        model = Z_in
    elif modelname == "Cstray":
        model = Z_stray
    else:
        raise NotImplementedError("Model {} not implemented".format(modelname))
    return model


def process_parallel(model_list):
    first = []
    second = []
    tag_first = True
    for m in model_list:
        if ',' not in m and tag_first:
            first.append(m)
        elif ',' not in m and tag_first is False:
            second.append(m)
        else:
            tmp = m.split(',')
            if len(tmp) != 2:
                raise RuntimeError("This should not be it! Parallel split looked like {}.".format(tmp))
            first.append(tmp[0])
            second.append(tmp[1])
            tag_first = False
    first_models = []
    second_models = []
    for m in first:
        m = m.replace("(", "")
        first_models.append(generate_model(m))
    for m in second:
        m = m.replace(")", "")
        second_models.append(generate_model(m))
    logger.debug("first_models: " + str(first_models))
    logger.debug("second_models: " + str(second_models))
    first = first_models[0]
    first_models.pop(0)
    for m in first_models:
        first += m
    second = second_models[0]
    second_models.pop(0)
    for m in second_models:
        second += m
    return CompositeModel(first, second, parallel)


def generate_model(m):
    if '_' in m:
        # get possible prefix
        info = m.split("_")
        if len(info) != 2:
            raise RuntimeError("A model must be always named PREFIX_MODEL! Instead you named it: {}".format(m))
        # also checks if model exists
        return Model(model_function(info[0].strip()), prefix=info[1].strip() + str('_'))
    else:
        return Model(model_function(m.strip()))


def get_comp_model(modelname):
    """
    get composite model
    """
    models = modelname.split("parallel")
    models = [m.strip().split('+') for m in models]
    # filter spaces out of list
    models = [list(filter(('').__ne__, m)) for m in models]
    compound = []
    logger.debug("Models: {}".format(models))
    for model in models:
        # iterate through groups
        logger.debug("Running model {}".format(model))
        for idx, m in enumerate(model):
            if '(' not in m:
                compound.append(generate_model(m))
            elif '(' in m:
                compound.append(process_parallel(model))
                break
            else:
                raise RuntimeError("There must have been a typo in your model specification.")
    if len(compound) == 0:
        raise RuntimeError("Did you provide an empty model?")
    composite_model = compound[0]
    compound.pop(0)
    for m in compound:
        composite_model += m
    logger.debug("Created composite model {}".format(composite_model))
    return composite_model
