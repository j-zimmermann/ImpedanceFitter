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


import matplotlib.pyplot as plt
import numpy as np
import yaml
import logging
from scipy.constants import epsilon_0 as e0
from collections import Counter
from lmfit import Parameters
from .elements import Z_C

logger = logging.getLogger('impedancefitter-logger')


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
        Zstray = Z_C(omega, cf)

    return (Zdut * Zstray) / (Zdut + Zstray)


def compare_to_data(omega, Z, Z_fit, subplot=None, title="", show=True, save=False):
    '''
    plots the relative difference of the fitted function to the data

    Parameters
    ----------
    omega: double or ndarray of double
        frequency array
    Z: complex or array of complex
        impedance array, experimental data or data to compare to
    Z_fit: complex or array of complex
        impedance array, fit result
    subplot: optional
        decide whether it is a new figure or a subplot. Default is None (yields new figure). Otherwise it can be an integer to denote the subfigure.
    title: str, optional
        title of plot. Default is an empty string.
    show: bool, optional
        show figure (default is True). Only has an effect when `subplot` is None.
    save: bool, optional
        save figure to pdf (default is False). Name of figure starts with `title`.
    '''
    if subplot is None:
        plt.figure()
    else:
        show = False
        plt.subplot(subplot)
    plt.xscale('log')
    plt.title(str(title) + "relative difference to data")
    plt.xlabel('frequency [Hz]')
    plt.ylabel('rel. difference to data [%]')
    plt.plot(omega / (2. * np.pi), 100. * np.abs((Z.real - Z_fit.real) / Z.real), 'g', label='rel .difference real part')
    plt.plot(omega / (2. * np.pi), 100. * np.abs((Z.imag - Z_fit.imag) / Z.imag), 'r', label='rel. difference imag part')
    plt.plot(omega / (2. * np.pi), 100. * np.abs((Z - Z_fit) / Z), 'b', label='rel. difference absolute value')
    plt.legend()
    if save:
        plt.savefig(str(title) + "_relative difference to data.pdf")
    if show:
        plt.show()


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


def plot_dielectric_properties(omega, Z, c0, Z_comp=None, title="", show=True, save=False):
    '''
    Parameters
    ----------

    omega: double or ndarray of double
        frequency array
    Z: complex or array of complex
        impedance array
    c0: double
        unit capacitance of device
    Z_comp: optional
        complex-valued impedance array. Might be used to compare the properties of two data sets.
    title: str, optional
        title of plot. Default is an empty string.
    show: bool, optional
        show figure (default is True)
    save: bool, optional
        save figure to pdf (default is False). Name of figure starts with `title`.

    '''
    eps_r, cond_fit = return_diel_properties(omega, Z, c0)
    if Z_comp is not None:
        eps_r2, cond_fit2 = return_diel_properties(omega, Z_comp, c0)
    plt.figure()
    plt.suptitle('dielectric properties', y=1.05)
    plt.subplot(211)
    plt.title("relative permittivity")
    plt.ylabel("relative permittivity")
    plt.xlabel('frequency [Hz]')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(omega / (2. * np.pi), eps_r, label="Z_1")
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), eps_r2, label="Z_2")
        plt.legend()

    plt.subplot(212)
    plt.title("conductivity")
    plt.ylabel("conductivity [S/m]")
    plt.xlabel('frequency [Hz]')
    plt.xscale('log')
    plt.plot(omega / (2. * np.pi), cond_fit, label="Z_1")
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), cond_fit2, label="Z_2")
        plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(str(title) + "_dielectric_properties.pdf")
    if show:
        plt.show()


def set_parameters(modelName, parameterdict=None, ep_cpe=False, ind=False, loss=False, stray=False):
    """
    Parameters
    -----------

    modelName: string
        name of the model. must be one of those listed in :func:`available_models`
    parameterdict: optional
        a dictionary containing parameters for model with *min*, *max*, *vary* info for LMFIT.
    ep_cpe: bool, optional
        switch on electrode polarization correction by CPE
    ind: bool, optional
        switch on correction of inductive effects by LR model
    loss: bool, optional
        switch on correction of inductive effects by LCR model for high loss materials
    stray: bool, optional
        switch on stray capacitance


    Returns:
    --------

    params: Parameters
        LMFIT `Parameters`.
    """

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
    # create empty lmfit parameters
    params = Parameters()
    bufdict = _clean_parameters(bufdict, modelName, ep_cpe, ind, loss)
    for key in parameter_names(modelName, ep_cpe=ep_cpe, ind=ind, loss=loss, stray=stray):
        params.add(key, value=float(bufdict[key]['value']))
        if 'min' in bufdict[key]:
            params[key].set(min=float(bufdict[key]['min']))
        if 'max' in bufdict[key]:
            params[key].set(max=float(bufdict[key]['max']))
        if 'vary' in bufdict[key]:
            params[key].set(vary=bool(bufdict[key]['vary']))
        if 'expr' in bufdict[key]:
            params[key].set(expr=str(bufdict[key]['vary']))
    return params


def _clean_parameters(params, modelName, ep, ind, loss):
    """
    clean parameter dicts that are passed to the fitter.
    get rid of parameters that are not needed.

    Parameters
    ----------
    params: dict
        input dictionary
    modelName: string
        name of model corresponding to the ones listed
    """
    names = parameter_names(modelName, ep, ind, loss)
    for p in list(params.keys()):
        if p not in names:
            del params[p]
    assert Counter(names) == Counter(params.keys()), "You need to provide the following parameters " + str(names)
    return params


def parameter_names(model, ep_cpe=False, ind=False, loss=False, stray=False):
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
    if model == 'cole_cole':
        names = ['c0', 'epsi_l', 'tau', 'a', 'conductivity', 'eh']
    elif model == 'cole_cole_r':
        names = ['Rinf', 'R0', 'tau', 'a']
    elif model == 'randles':
        names = ['Rinf', 'R0', 'tau', 'a']
    elif model == 'randles_cpe':
        names = ['Rinf', 'R0', 'tau', 'a']
    elif model == 'rc':
        names = ['c0', 'conductivity', 'eps']
    elif model == 'RC':
        names = ['Rd', 'Cd']
    elif model == 'suspension':
        names = ['cf', 'epsi_l', 'tau', 'a', 'conductivity', 'eh']
    elif model == 'single_shell':
        names = ['c0', 'em', 'km', 'kcp', 'ecp', 'kmed', 'emed', 'p', 'dm', 'Rc']
    elif model == 'double_shell':
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
        'epsi_l': r'$\varepsilon_\mathrm{l}$',
        'tau': r'$\tau$',
        'a': r'$a$',
        'alpha': r'$\alpha$',
        'conductivity': r'$\sigma_\mathrm{DC}$',
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


def plot_results(omega, Z, Z_fit, title, show=True, save=False):
    """
    Plot the `result` and compare it to data `Z`.
    Generates 4 subplots showing the real and imaginary parts over
    the frequency; a Nyquist plot of real and negative imaginary part
    and the relative differences of real and imaginary part as well as absolute value of impedance.

    Parameters
    ----------
    omega: double or ndarray of double
        frequency array
    Z: complex or array of complex
        impedance array, experimental data or data to compare to
    Z_fit: complex or array of complex
        impedance array, fit result
    title: str
        title of plot
    show: bool, optional
        show figure (default is True)
    save: bool, optional
        save figure to pdf (default is False). Name of figure starts with `title`.

    """

    plt.figure()
    plt.suptitle(title, y=1.05)
    # plot real part of impedance
    plt.subplot(221)
    plt.xscale('log')
    plt.title("impedance real part")
    plt.ylabel(r"$\Re(Z) [\Omega]$")
    plt.xlabel("frequency [Hz]")
    plt.plot(omega / (2. * np.pi), Z_fit.real, '+', label='fitted')
    plt.plot(omega / (2. * np.pi), Z.real, 'r', label='data')
    plt.legend()
    # plot imaginary part of impedance
    plt.subplot(222)
    plt.title("impedance imaginary part")
    plt.xscale('log')
    plt.ylabel(r"$\Im(Z) [\Omega]$")
    plt.xlabel("frequency [Hz]")
    plt.plot(omega / (2. * np.pi), Z_fit.imag, '+', label='fitted')
    plt.plot(omega / (2. * np.pi), Z.imag, 'r', label='data')
    plt.legend()
    # plot real vs negative imaginary part
    plt.subplot(223)
    plt.title("Nyquist plot")
    plt.ylabel(r"$-\Im(Z) [\Omega]$")
    plt.xlabel(r"$\Re(Z) [\Omega]$")
    plt.plot(Z_fit.real, Z_fit.imag, '+', label="fit")
    plt.plot(Z.real, Z.imag, 'o', label="data")
    plt.legend()
    compare_to_data(omega, Z, Z_fit, title, subplot=224)
    plt.tight_layout()
    if save:
        plt.savefig(str(title) + "_results_overview.pdf")
    if show:
        plt.show()


def available_models():
    """
    return list of available models
    """
    models = ['cole_cole',
              'cole_cole_r',
              'randles',
              'randles_cpe',
              'rc',
              'RC',
              'suspension',
              'single_shell',
              'double_shell']
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
    information = {'cole_cole': """ Cole-Cole model as implemented in paper with DOI:10.1063/1.4737121.
                                    You need to provide the unit capacitance of your device to get the dielectric properties of
                                    the Cole-Cole model. """,
                   'cole_cole_r': """ Standard Cole-Cole circuit as given in paper with DOI:10.1016/b978-1-4832-3111-2.50008-0.
                                      Implemented in :func:impedancefitter.cole_cole_R.co """,
                   'randles': """
                                """,
                   'randles_cpe': """
                                """,
                   'rc': """
                   """,
                   'RC': """
                   """,
                   'suspension': """
                   """,
                   'single_shell': """
                   """,
                   'double_shell': """
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
