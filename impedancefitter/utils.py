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


import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.constants import epsilon_0 as e0
import logging
from collections import Counter


logger = logging.getLogger('impedancefitter-logger')


def Z_loss(omega, L, C, R):
    """
    impedance for high loss materials
    """
    Y = 1. / R + 1. / (1j * omega * L) + 1j * omega * C
    Z = 1. / Y
    return Z


def Z_in(omega, L, R):
    """
    Lead inductance
    """
    return R + 1j * omega * L


def Z_CPE(omega, k, alpha):
    """
    CPE impedance
    """
    return (1. / k) * (1j * omega) ** (-alpha)


def Z_sus(omega, es, kdc, c0, cf):
    """
    Accounts for air capacitance and stray capacitance.
    """
    return 1. / (1j * es * omega * c0 + 1j * (kdc * c0) / e0 + 1j * omega * cf)


def e_sus(omega, eh, el, tau, a):  # this is only valid for the cole_cole_fit and suspension_fit
    """
    complex permitivity of suspension
    """
    return eh + (el - eh) / (1. + (1j * omega * tau) ** (a))


def compare_to_data(omega, Z, Z_fit, filename, subplot=None):
    '''
    plots the relative difference of the fitted function to the data
    '''
    if subplot is None:
        plt.figure()
    else:
        plt.subplot(subplot)
    plt.xscale('log')
    if subplot is None:
        plt.title(str(filename) + "relative difference to data")
    else:
        plt.title("relative difference to data")
    plt.xlabel('frequency [Hz]')
    plt.ylabel('rel. difference [%] to data')
    plt.plot(omega / (2. * np.pi), 100. * np.abs((Z.real - Z_fit.real) / Z.real), 'g', label='rel .difference real part')
    plt.plot(omega / (2. * np.pi), 100. * np.abs((Z.imag - Z_fit.imag) / Z.imag), 'r', label='rel. difference imag part')
    plt.plot(omega / (2. * np.pi), 100. * np.abs((Z - Z_fit) / Z), 'b', label='rel. difference absolute value')
    plt.legend()


def return_diel_properties(omega, epsc):
    """
    return real permittivity and conductivity
    """
    eps_r = epsc.real
    conductivity = -epsc.imag * e0 * omega
    return eps_r, conductivity


def plot_dielectric_properties(omega, cole_cole_output, suspension_output):
    '''
    eps is the complex valued permitivity from which we extract relative permitivity and conductivity
    compares the dielectric properties before and after the compensation
    '''
    eh = suspension_output.params.valuesdict()['eh']
    el = suspension_output.params.valuesdict()['epsi_l']
    tau = suspension_output.params.valuesdict()['tau']
    a = suspension_output.params.valuesdict()['a']
    eps_fit = e_sus(omega, eh, el, tau, a)
    eps_r_fit, cond_fit = return_diel_properties(omega, eps_fit)
    eh = cole_cole_output.params.valuesdict()['eh']
    el = cole_cole_output.params.valuesdict()['epsi_l']
    tau = cole_cole_output.params.valuesdict()['tau']
    a = cole_cole_output.params.valuesdict()['a']
    eps_fit_corrected = e_sus(omega, eh, el, tau, a)
    eps_r_fit_corr, cond_fit_corr = return_diel_properties(omega, eps_fit_corrected)
    plt.figure()
    plt.suptitle('dielectric properties compared', y=1.05)
    plt.subplot(211)
    plt.title("permittivity")
    plt.ylabel("relative permittivity")
    plt.xlabel('frequency [Hz]')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(omega / (2. * np.pi), eps_r_fit, label="fit")
    plt.plot(omega / (2. * np.pi), eps_r_fit_corr, label="corrected")
    plt.legend()

    plt.subplot(212)
    plt.title("conductivity")
    plt.ylabel("conductivity [S/m]")
    plt.xlabel('frequency [Hz]')
    plt.xscale('log')
    plt.plot(omega / (2. * np.pi), cond_fit, label="fit")
    plt.plot(omega / (2. * np.pi), cond_fit_corr, label="corrected")
    plt.legend()
    plt.tight_layout()
    plt.show()


def set_parameters(params, modelName, parameterdict, ep=False, ind=False, loss=False):
    """
    for suspension model: if wanted one could create an own file,
    otherwise the value from the cole-cole model are taken.

    Parameter `ep` is False if no electrode polarization is used.

    The parameters are returned in a strict order!
    """

    if parameterdict is None:
        try:
            infile = open(modelName + '_input.yaml', 'r')
            bufdict = yaml.safe_load(infile)
        except FileNotFoundError as e:
            if(modelName == 'suspension'):
                infile = open('cole_cole_input.yaml', 'r')
                bufdict = yaml.safe_load(infile)
                if 'alpha' in bufdict:
                    del bufdict['alpha']
                if 'k' in bufdict:
                    del bufdict['k']
            else:
                print(str(e))
                print("Please provide a yaml-input file.")
                raise
    else:
        try:
            bufdict = parameterdict[modelName]
        except KeyError:
            if(modelName == 'suspension'):
                bufdict = parameterdict['cole_cole']
                if 'alpha' in bufdict:
                    del bufdict['alpha']
                if 'k' in bufdict:
                    del bufdict['k']
                pass
            else:
                print("Your parameterdict lacks an entry for the model: " + modelName)
                raise
    bufdict = clean_parameters(bufdict, modelName, ep, ind, loss)
    for key in parameter_names(modelName, ep, ind=ind, loss=loss):
        params.add(key, value=float(bufdict[key]['value']))
        if 'min' in bufdict[key]:
            params[key].set(min=float(bufdict[key]['min']))
        if 'max' in bufdict[key]:
            params[key].set(max=float(bufdict[key]['max']))
        if 'vary' in bufdict[key]:
            params[key].set(vary=bool(bufdict[key]['vary']))
    return params


def clean_parameters(params, modelName, ep, ind, loss):
    names = parameter_names(modelName, ep, ind, loss)
    for p in list(params.keys()):
        if p not in names:
            del params[p]
    assert Counter(names) == Counter(params.keys()), "You need to provide the following parameters " + str(names)
    return params


def parameter_names(model, ep, ind=False, loss=False):
    """
    Get the order of parameters for a certain model.

    Takes model as a string and ep as a bool (switch electrode polarisation on or off).
    """
    if model == 'cole_cole':
        names = ['c0', 'cf', 'epsi_l', 'tau', 'a', 'conductivity', 'eh']
    elif model == 'suspension':
        names = ['c0', 'cf', 'epsi_l', 'tau', 'a', 'conductivity', 'eh']
    elif model == 'single_shell':
        names = ['c0', 'cf', 'em', 'km', 'kcp', 'ecp', 'kmed', 'emed', 'p', 'dm', 'Rc']
    elif model == 'double_shell':
        names = ['c0', 'cf', 'em', 'km', 'kcp', 'ecp', 'kmed', 'emed',
                 'p', 'dm', 'Rc', 'ene', 'kne', 'knp', 'enp', 'dn', 'Rn']
    if ep is True:
        names.extend(['k', 'alpha'])
    if ind is True:
        names.extend(['L', 'R'])
    if loss is True:
        names.extend(['L', 'C', 'R'])
    return names


def get_labels():
    labels = {
        'c0': r'$C_0$',
        'cf': r'$C_\mathrm{f}',
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
        'R': r'$R$'
        }
    return labels
