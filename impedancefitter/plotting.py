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
import corner

from .utils import return_diel_properties, get_labels


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
    plot_compare_to_data(omega, Z, Z_fit, subplot=224)
    plt.tight_layout()
    if save:
        plt.savefig(str(title) + "_results_overview.pdf")
    if show:
        plt.show()


def plot_compare_to_data(omega, Z, Z_fit, subplot=None, title="", show=True, save=False):
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


def emcee_plot(res, **corner_kwargs):
    """
    Create corner plot.

    Parameters
    ----------
    res: dict
        Dictionary containing values and flatchain
    corner_kwargs: dict, optional
        Dictionary with further corner plot options
    """
    truths = [res.params[p].value for p in res.var_names]
    ifLabels = get_labels()
    labels = [ifLabels[p] for p in res.var_names]
    plot = corner.corner(res.flatchain, labels=labels,
                         truths=truths, **corner_kwargs)
    return plot
