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
from .elements import Z_C, Z_CPE, Z_in, Z_loss

logger = logging.getLogger('impedancefitter-logger')


def add_additions(omega, Zs_fit, k, alpha, L, C, R, cf):
    """
    .. todo::
        documentation
    """

    # add first CPE
    if k is not None and alpha is not None:
        Zep_fit = Z_CPE(omega, k, alpha)
        Zs_fit = Zs_fit + Zep_fit

    # then stray capacitance, which is always in parallel to input
    # and electrode polarization impedance
    if cf is not None:
        Zs_fit = add_stray_capacitance(omega, Zs_fit, cf)

    # then add the influence of the wiring
    if L is not None:
        if C is None:
            Zin_fit = Z_in(omega, L, R)
        elif C is not None and R is not None:
            Zin_fit = Z_loss(omega, L, C, R)
        Zs_fit = Zs_fit + Zin_fit

    return Zs_fit


def add_stray_capacitance(omega, Zdut, cf):
    r"""
    add stray capacitance to impedance.
    The assumption made is that the stray capacitance is in parallel to the DUT.
    Hence, the total impedance is

    .. math::

        Z_\mathrm{total} = \left(\frac{1}{Z_\mathrm{DUT}} + \frac{1}{Z_\mathrm{stray}}\right)^{-1}

    .. note::
        The stray capacitance is always given in pF for numerical reasons!
        It is checked if the stray capacitance is greater than 0 with an absolute tolerance of :math:`10^{-5}` pF.

    Parameters
    ----------
    omega: double or ndarray of double
        frequency array
    Zdut: complex or array of complex
        impedance array, data of DUT
    cf: double
        stray capacitance to be accounted for
    """

    if np.isclose(cf, 0, atol=1e-5):
        logger.debug("""Stray capacitance is too small to be added.
                     Did you maybe forget to enter it in terms of pF?""")
        return Zdut
    else:
        cf *= 1e-12
        Zstray = Z_C(omega, cf)

    return (Zdut * Zstray) / (Zdut + Zstray)


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


def set_parameters(model, parameterdict=None, emcee=False):
    """
    Parameters
    -----------

    model: string
        LMFIT model object.
    parameterdict: optional
        a dictionary containing parameters for model with *min*, *max*, *vary* info for LMFIT.
        If it is None (default), the parameters are read in from a yaml-file.
    ep_cpe: bool, optional
        switch on electrode polarization correction by CPE
    ind: bool, optional
        switch on correction of inductive effects by LR model
    loss: bool, optional
        switch on correction of inductive effects by LCR model for high loss materials
    stray: bool, optional
        switch on stray capacitance
    emcee: bool, optional
        if emcee is used, an additional `__lnsigma` parameter will be set


    Returns:
    --------

    params: Parameters
        LMFIT `Parameters`.
    """
    modelName = model._name
    if parameterdict is None:
        try:
            infile = open(modelName + '_input.yaml', 'r')
            bufdict = yaml.safe_load(infile)
        except OSError:
            logger.error("Please provide a yaml-input file for model: ", modelName)
    else:
        try:
            bufdict = parameterdict[modelName]
        except KeyError:
            logger.error("Your parameterdict lacks an entry for the model: " + modelName)

    bufdict = _clean_parameters(bufdict, modelName, model.param_names, emcee)
    for key in model.param_names:
        model.set_param_hint(key, **bufdict[key])
    return model.make_params()


def _clean_parameters(params, modelName, names, emcee):
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
    if emcee and '__lnsigma' not in params:
        logger.warning("""You did not provide the parameter '__lnsigma'.
                          It is needed for the emcee of unweighted data (as implemented here).
                          Use for example the lmfit default:\n
                          value=np.log(0.1), min=np.log(0.001), max=np.log(2)""")
    assert Counter(names) == Counter(params.keys()), "You need to provide the following parameters " + str(names)
    return params


def parameter_names(model, ep_cpe=False, ind=False, loss=False, stray=False, emcee=False):
    """
    Get the right order of parameters for a certain model.
    Is needed to pass the parameters properly to each model function call.

    Parameters
    ----------
    model: string
        name of the model
    ep_cpe: bool, optional
        switch on electrode polarization correction by CPE
    ind: bool, optional
        switch on correction of inductive effects by LR model
    loss: bool, optional
        switch on correction of inductive effects by LCR model for high loss materials
    stray: bool, optional
        switch on stray capacitance

    Returns
    -------
    names: list
        List of parameters needed for model

    """
    if model == 'ColeCole':
        names = ['c0', 'el', 'tau', 'a', 'kdc', 'eh']
    elif model == 'ColeColeR':
        names = ['Rinf', 'R0', 'tau', 'a']
    elif model == 'Randles':
        names = ['Rinf', 'R0', 'tau', 'a']
    elif model == 'Randles_CPE':
        names = ['Rinf', 'R0', 'tau', 'a']
    elif model == 'RC':
        names = ['c0', 'conductivity', 'eps']
    elif model == 'RC_full':
        names = ['Rd', 'Cd']
    elif model == 'SingleShell':
        names = ['c0', 'em', 'km', 'kcp', 'ecp', 'kmed', 'emed', 'p', 'dm', 'Rc']
    elif model == 'DoubleShell':
        names = ['c0', 'em', 'km', 'kcp', 'ecp', 'kmed', 'emed',
                 'p', 'dm', 'Rc', 'ene', 'kne', 'knp', 'enp', 'dn', 'Rn']
    else:
        raise NotImplementedError("Model not known!")
    if ep_cpe is True:
        names.extend(['k', 'alpha'])
    if ind is True:
        names.extend(['L', 'R'])
    if loss is True:
        names.extend(['L', 'C', 'R'])
    if stray is True:
        names.extend(['cf'])
    if emcee is True:
        names.extend(['__lnsigma'])
    return names


def get_labels():
    """
    return the labels for every parameter in LaTex code.

    Returns
    -------

    labels: dict
        dictionary with parameter names as keys and LaTex code as values.
    """

    labels = {
        'c0': r'$C_0$',
        'cf': r'$C_\mathrm{f}$',
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
        'eps': r'$\varepsilon_\mathrm{r}$'}

    return labels


def available_models():
    """
    return list of available models
    """
    models = ['ColeCole',
              'ColeColeR',
              'Randles',
              'Randles_CPE',
              'RC_full',
              'RC',
              'SingleShell',
              'DoubleShell']
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
                   'Randles_CPE': """
                                """,
                   'RC_full': """
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
