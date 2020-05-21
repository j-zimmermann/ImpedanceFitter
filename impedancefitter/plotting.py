#    The ImpedanceFitter is a package to fit impedance spectra to equivalent-circuit models using open-source software.
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


def plot_bode(omega, Z, title="", Z_fit=None, show=True, save=False, Z_comp=None,
              labels=["data", "best fit", "init fit"]):
    """Bode plot of impedance.

    Plots phase and log of magnitude over log of frequency.

    Parameters
    ----------
    omega: double or ndarray of double
        Frequency array
    Z: complex or array of complex
        Impedance array, experimental data or data to compare to.
    Z_fit: complex or array of complex
        Impedance array, fit result. If provided, the difference
        between data and fit will be shown.
    title: str
        Title of plot.
    show: bool, optional
        Show figure (default is True).
    save: bool, optional
        Save figure to pdf (default is False). Name of figure starts with `title`.
    Z_comp: optional
        Complex-valued impedance array. Might be used to compare the properties of two data sets.
    labels: list
        List of labels for three plots. Must have length 3 always.
        Is ordered like: `[Z, Z_fit, Z_comp]`
    """

    plt.figure()
    plt.suptitle(title, y=1.05)
    # plot real part of impedance
    plt.subplot(211)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r"|Z| $\Omega$")
    plt.xlabel("frequency [Hz]")
    plt.plot(omega / (2. * np.pi), np.abs(Z), 'r', label=labels[0])
    if Z_fit is not None:
        plt.plot(omega / (2. * np.pi), np.abs(Z_fit), '+', label=labels[1])
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), np.abs(Z_comp), 'x', label=labels[2])
    plt.legend()
    plt.subplot(212)
    plt.xscale('log')
    plt.ylabel("phase [°]")
    plt.xlabel("frequency [Hz]")
    plt.plot(omega / (2. * np.pi), np.angle(Z, deg=True), 'r', label=labels[0])
    if Z_fit is not None:
        plt.plot(omega / (2. * np.pi), np.angle(Z_fit, deg=True), '+', label=labels[1])
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), np.angle(Z_comp, deg=True), 'x', label=labels[2])
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(str(title) + "_bode_plot.pdf")
    if show:
        plt.show()


def plot_impedance(omega, Z, title="", Z_fit=None, show=True, save=False, Z_comp=None,
                   labels=["data", "best fit", "init fit"], relative=True, sign=False):
    """Plot the `result` and compare it to data `Z`.

    Generates 4 subplots showing the real and imaginary parts over
    the frequency; a Nyquist plot of real and negative imaginary part
    and the relative differences of real and imaginary part as well as absolute value of impedance.

    Parameters
    ----------
    omega: double or ndarray of double
        Frequency array
    Z: complex or array of complex
        Impedance array, experimental data or data to compare to.
    Z_fit: complex or array of complex
        Impedance array, fit result. If provided, the difference
        between data and fit will be shown.
    title: str
        Title of plot.
    show: bool, optional
        Show figure (default is True).
    save: bool, optional
        Save figure to pdf (default is False). Name of figure starts with `title`.
    Z_comp: optional
        Complex-valued impedance array. Might be used to compare the properties of two data sets.
    labels: list
        List of labels for three plots. Must have length 3 always.
        Is ordered like: `[Z, Z_fit, Z_comp]`
    relative: bool
        Plot relative difference if True, else plot residual.
    sign: bool, optional
        Use sign of residual. Default is False, i.e. absolute value is plotted.

    """

    plt.figure()
    plt.suptitle(title, y=1.05)
    # plot real part of impedance
    plt.subplot(221)
    plt.xscale('log')
    plt.title("impedance real part")
    plt.ylabel(r"$\Re(Z) [\Omega]$")
    plt.xlabel("frequency [Hz]")
    plt.plot(omega / (2. * np.pi), Z.real, 'r', label=labels[0])
    if Z_fit is not None:
        plt.plot(omega / (2. * np.pi), Z_fit.real, '+', label=labels[1])
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), Z_comp.real, 'x', label=labels[2])
    plt.legend()
    # plot imaginary part of impedance
    plt.subplot(222)
    plt.title("impedance imaginary part")
    plt.xscale('log')
    plt.ylabel(r"$\Im(Z) [\Omega]$")
    plt.xlabel("frequency [Hz]")
    plt.plot(omega / (2. * np.pi), Z.imag, 'r', label=labels[0])
    if Z_fit is not None:
        plt.plot(omega / (2. * np.pi), Z_fit.imag, '+', label=labels[1])
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), Z_comp.imag, 'x', label=labels[2])
    plt.legend()
    # plot real vs negative imaginary part
    plt.subplot(223)
    plt.title("Nyquist plot")
    plt.ylabel(r"$-\Im(Z) [\Omega]$")
    plt.xlabel(r"$\Re(Z) [\Omega]$")
    plt.plot(Z.real, -Z.imag, 'o', label=labels[0])
    if Z_fit is not None:
        plt.plot(Z_fit.real, -Z_fit.imag, '+', label=labels[1])
    if Z_comp is not None:
        plt.plot(Z_comp.real, -Z_comp.imag, 'x', label=labels[2])
    plt.legend()
    if Z_fit is not None:
        plot_compare_to_data(omega, Z, Z_fit, subplot=224, relative=True, sign=sign)
    plt.tight_layout()
    if save:
        plt.savefig(str(title) + "_results_overview.pdf")
    if show:
        plt.show()


def plot_compare_to_data(omega, Z, Z_fit, subplot=None, title="", show=True, save=False,
                         relative=True, sign=False):
    '''
    plots the difference of the fitted function to the data

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
    relative: bool, optional
        Plot relative difference if True, else plot residual.
    sign: bool, optional
        Use sign of residual. Default is False, i.e. absolute value is plotted.
    '''
    if subplot is None:
        plt.figure()
    else:
        show = False
        plt.subplot(subplot)
    if relative:
        diff_real = 100. * (Z.real - Z_fit.real) / Z.real
        diff_imag = 100. * (Z.imag - Z_fit.imag) / Z.imag
        diff_abs = 100. * np.abs((Z - Z_fit) / Z)
        lbl_prefix = 'rel. '
    else:
        diff_real = Z.real - Z_fit.real
        diff_imag = Z.imag - Z_fit.imag
        diff_abs = np.abs(Z - Z_fit)
        lbl_prefix = ''
    if not sign:
        diff_real = np.abs(diff_real)
        diff_imag = np.abs(diff_imag)

    plt.xscale('log')
    plt.title(str(title) + "relative difference to data")
    plt.xlabel('frequency [Hz]')
    plt.ylabel('rel. difference to data [%]')
    plt.plot(omega / (2. * np.pi), diff_real, 'g', label=lbl_prefix + 'difference real part')
    plt.plot(omega / (2. * np.pi), diff_imag, 'r', label=lbl_prefix + 'difference imag part')
    plt.plot(omega / (2. * np.pi), diff_abs, 'b', label=lbl_prefix + 'difference absolute value')
    plt.legend()
    if save:
        if relative:
            plt.savefig(str(title) + "_relative_difference_to_data.pdf")
        else:
            plt.savefig(str(title) + "_difference_to_data.pdf")
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
    params = [p for p in res.params if res.params[p].vary]
    truths = [res.params[p].value for p in params]
    ifLabels = get_labels(params)
    labels = [ifLabels[p] for p in res.var_names]
    plot = corner.corner(res.flatchain, labels=labels,
                         truths=truths, **corner_kwargs)
    return plot


def plot_uncertainty(omega, Zdata, Z, Z1, Z2, sigma, show=True, model=None):
    """
    .. todo::
        documentation
    Parameters
    ----------

    model: int, optional
        numbering of model for sequential plotting

    """
    plt.figure()
    if model is not None:
        plt.suptitle(r"${} \sigma$ results".format(sigma), y=1.05)
    else:
        plt.suptitle(r"${} \sigma$ results, model {}".format(sigma, model), y=1.05)
    # plot real part of impedance
    plt.subplot(211)
    plt.xscale('log')
    plt.title("impedance real part")
    plt.ylabel(r"$\Re(Z) [\Omega]$")
    plt.xlabel("frequency [Hz]")
    plt.plot(omega / (2. * np.pi), Z.real, '+', label='best fit')
    plt.plot(omega / (2. * np.pi), Zdata.real, 'r', label='data')
    plt.fill_between(omega / (2. * np.pi), Z1.real, Z2.real,
                     color='#888888', label=r"${} \sigma$ interval".format(sigma))
    plt.legend()
    # plot imaginary part of impedance
    plt.subplot(212)
    plt.title("impedance imaginary part")
    plt.xscale('log')
    plt.ylabel(r"$\Im(Z) [\Omega]$")
    plt.xlabel("frequency [Hz]")
    plt.plot(omega / (2. * np.pi), Z.imag, '+', label='best fit')
    plt.plot(omega / (2. * np.pi), Zdata.imag, 'r', label='data')
    plt.fill_between(omega / (2. * np.pi), Z1.imag, Z2.imag,
                     color='#888888', label=r"${} \sigma$ interval".format(sigma))
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
