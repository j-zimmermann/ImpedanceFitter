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

from .utils import return_diel_properties, get_labels, _return_resistance_capacitance, return_dielectric_modulus
from scipy.constants import epsilon_0 as e0

import logging
logger = logging.getLogger(__name__)


def plot_complex_permittivity(omega, Z, c0, Z_comp=None,
                              title="", show=True, save=False,
                              logscale="permittivity", labels=None):
    '''
    Parameters
    ----------

    omega: :class:`numpy.ndarray`, double
        frequency array
    Z: :class:`numpy.ndarray`, complex
        impedance array
    c0: double
        unit capacitance of device
    Z_comp: :class:`numpy.ndarray`, complex, optional
        complex-valued impedance array. Might be used to compare the properties of two data sets.
    title: str, optional
        title of plot. Default is an empty string.
    show: bool, optional
        show figure (default is True)
    save: bool, optional
        save figure to pdf (default is False). Name of figure starts with `title`
        and ends with `_dielectric_properties.pdf`.
    logscale: str, optional
        Decide what you want to plot using log scale.
        Possible are `permittivity`, `loss` and `both`
    labels: list, optional
        Give custom labels. Needs to be a list of length 2.
    '''
    eps_r, cond_fit = return_diel_properties(omega, Z, c0)
    if labels is None:
        labels = [r'$Z_1$', r'$Z_2$']
    assert len(labels) == 2, "You need to provide lables as a list containing 2 strings!"
    if Z_comp is not None:
        eps_r2, cond_fit2 = return_diel_properties(omega, Z_comp, c0)
    plt.figure()
    plt.suptitle('complex permittivity', y=1.05)
    plt.subplot(211)
    plt.title("Relative permittivity")
    plt.ylabel("Relative permittivity")
    plt.xlabel('Frequency / Hz')
    if logscale == 'permittivity' or logscale == 'both':
        plt.yscale('log')
    plt.xscale('log')
    plt.plot(omega / (2. * np.pi), eps_r, label=labels[0])
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), eps_r2, label=labels[1])
        plt.legend()

    plt.subplot(212)
    plt.title("Dielectric loss")
    plt.ylabel("Dielectric loss")
    plt.xlabel('Frequency / Hz')
    if logscale == 'loss' or logscale == 'both':
        plt.yscale('log')
    plt.xscale('log')
    plt.plot(omega / (2. * np.pi), cond_fit / (e0 * omega), label=labels[0])
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), cond_fit2 / (e0 * omega), label=labels[1])
        plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(str(title).replace(" ", "_") + "_complex_permittivity.pdf")
    if show:
        plt.show()
    else:
        plt.close()


def plot_dielectric_modulus(omega, Z, c0, Z_comp=None,
                            title="", show=True, save=False,
                            logscale=None, labels=None):
    '''
    Parameters
    ----------

    omega: :class:`numpy.ndarray`, double
        frequency array
    Z: :class:`numpy.ndarray`, complex
        impedance array
    c0: double
        unit capacitance of device
    Z_comp: :class:`numpy.ndarray`, complex, optional
        complex-valued impedance array. Might be used to compare the properties of two data sets.
    title: str, optional
        title of plot. Default is an empty string.
    show: bool, optional
        show figure (default is True)
    save: bool, optional
        save figure to pdf (default is False). Name of figure starts with `title`
        and ends with `_dielectric_properties.pdf`.
    logscale: str, optional
        Decide what you want to plot using log scale.
        Possible are `ReM`, `ImM` and `both`
    labels: list, optional
        Give custom labels. Needs to be a list of length 2.
    '''
    ReM, ImM = return_dielectric_modulus(omega, Z, c0)
    if labels is None:
        labels = [r'$Z_1$', r'$Z_2$']
    assert len(labels) == 2, "You need to provide lables as a list containing 2 strings!"
    if Z_comp is not None:
        ReM2, ImM2 = return_dielectric_modulus(omega, Z_comp, c0)
    plt.figure()
    plt.suptitle('Real part', y=1.05)
    plt.subplot(211)
    plt.ylabel("Re M")
    plt.xlabel('Frequency / Hz')
    if logscale == 'ReM' or logscale == 'both':
        plt.yscale('log')
    plt.xscale('log')
    plt.plot(omega / (2. * np.pi), ReM, label=labels[0])
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), ReM2, label=labels[1])
        plt.legend()

    plt.subplot(212)
    plt.title("Imaginary part")
    plt.ylabel("Im M")
    plt.xlabel('Frequency / Hz')
    if logscale == 'ImM' or logscale == 'both':
        plt.yscale('log')
    plt.xscale('log')
    plt.plot(omega / (2. * np.pi), ImM, label=labels[0])
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), ImM2, label=labels[1])
        plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(str(title).replace(" ", "_") + "_dielectric_modulus.pdf")
    if show:
        plt.show()
    else:
        plt.close()


def plot_dielectric_properties(omega, Z, c0, Z_comp=None, title="", show=True, save=False, logscale="permittivity",
                               labels=None, append=False, markers=[None, None], **plotkwargs):
    '''
    Parameters
    ----------

    omega: :class:`numpy.ndarray`, double
        frequency array
    Z: :class:`numpy.ndarray`, complex
        impedance array
    c0: double
        unit capacitance of device
    Z_comp: :class:`numpy.ndarray`, complex, optional
        complex-valued impedance array. Might be used to compare the properties of two data sets.
    title: str, optional
        title of plot. Default is an empty string.
    show: bool, optional
        show figure (default is True)
    save: bool, optional
        save figure to pdf (default is False). Name of figure starts with `title`
        and ends with `_dielectric_properties.pdf`.
    logscale: str, optional
        Decide what you want to plot using log scale.
        Possible are `permittivity`, `conductivity` and `both`
    labels: list, optional
        Give custom labels. Needs to be a list of length 2.
    append: bool, optional
        Decide if you want to show plot or add line to existing plot.

    '''
    eps_r, cond_fit = return_diel_properties(omega, Z, c0)

    axes = []
    plt.figure("dielectricproperties")
    axes = plt.gcf().axes

    if len(axes) < 2:
        plt.suptitle(title, y=1.05)
        plt.subplot(211)
    else:
        plt.sca(axes[0])

    if labels is None:
        labels = [r'$Z_1$', r'$Z_2$']
    assert len(labels) == 2, "You need to provide lables as a list containing 2 strings!"
    if Z_comp is not None:
        eps_r2, cond_fit2 = return_diel_properties(omega, Z_comp, c0)
    plt.title("Relative permittivity")
    plt.ylabel("Relative permittivity")
    plt.xlabel('Frequency / Hz')
    if logscale == 'permittivity' or logscale == 'both':
        plt.yscale('log')
    plt.xscale('log')
    plt.plot(omega / (2. * np.pi), eps_r, label=labels[0], marker=markers[0], **plotkwargs)
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), eps_r2, label=labels[1], marker=markers[1], **plotkwargs)
    if Z_comp is None and append is True:
        plt.legend()

    if len(axes) < 2:
        plt.subplot(212)
    else:
        plt.sca(axes[1])

    plt.title("Conductivity")
    plt.ylabel(r"Conductivity / S$\cdot$m$^{-1}$")
    plt.xlabel('Frequency / Hz')
    if logscale == 'conductivity' or logscale == 'both':
        plt.yscale('log')
    plt.xscale('log')
    plt.plot(omega / (2. * np.pi), cond_fit, label=labels[0], marker=markers[0], **plotkwargs)
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), cond_fit2, label=labels[1], marker=markers[1], **plotkwargs)
    if Z_comp is None and append is True:
        plt.legend()
    plt.tight_layout()
    if save and not append:
        plt.savefig(str(title).replace(" ", "_") + "_dielectric_properties.pdf")
    if show and not append:
        plt.show()
    elif not show and not append:
        plt.close()


def plot_cole_cole(omega, Z, c0, Z_comp=None, diff=False,
                   title="", show=True, save=False, labels=None):
    '''
    Parameters
    ----------

    omega: :class:`numpy.ndarray`, double
        frequency array
    Z: :class:`numpy.ndarray`, complex
        impedance array
    c0: double
        unit capacitance of device
    Z_comp: :class:`numpy.ndarray`, complex, optional
        complex-valued impedance array. Might be used to compare the properties of two data sets.
    title: str, optional
        title of plot. Default is an empty string.
    diff: bool, optional
        Compare two results numerically by plotting difference (default False)
    show: bool, optional
        show figure (default is True)
    save: bool, optional
        save figure to pdf (default is False). Name of figure starts with `title`
        and ends with `_dielectric_properties.pdf`.
    logscale: str, optional
        Decide what you want to plot using log scale.
        Possible are `permittivity`, `conductivity` and `both`
    labels: list, optional
        Give custom labels. Needs to be a list of length 2.
    '''
    eps_r, cond_fit = return_diel_properties(omega, Z, c0)
    epsc_fit = eps_r - 1j * cond_fit / (e0 * omega)
    if labels is None:
        labels = [r'$Z_1$', r'$Z_2$']
    assert len(labels) == 2, "You need to provide lables as a list containing 2 strings!"
    if Z_comp is not None:
        eps_r2, cond_fit2 = return_diel_properties(omega, Z_comp, c0)
        epsc_fit2 = eps_r2 - 1j * cond_fit2 / (e0 * omega)
        if diff:
            plt.figure()
            plt.subplot(211)

    plt.title("Cole-Cole plot")
    plt.xlabel(r"Re $\varepsilon$")
    plt.ylabel(r"-Im $\varepsilon$")
    plt.plot(epsc_fit.real, -epsc_fit.imag, label=labels[0])
    if Z_comp is not None:
        plt.plot(epsc_fit2.real, -epsc_fit2.imag, label=labels[1])
        plt.legend()

    if Z_comp is not None and diff:
        plt.subplot(212)
        plt.title("Comparison")
        plt.ylabel("Relative difference / %")
        plt.xlabel('Frequency / Hz')
        plt.xscale('log')
        plt.plot(omega / (2. * np.pi), 100. * np.abs((epsc_fit.real - epsc_fit2.real) / epsc_fit.real), label="real")
        plt.plot(omega / (2. * np.pi), 100. * np.abs((epsc_fit.imag - epsc_fit2.imag) / epsc_fit.imag), label="imag")
        plt.plot(omega / (2. * np.pi), 100. * np.abs((epsc_fit - epsc_fit2) / epsc_fit), label="abs")
        plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(str(title).replace(" ", "_") + "_cole_cole_plot.pdf")
    if show:
        plt.show()
    else:
        plt.close()


def plot_bode(omega, Z, title="", Z_fit=None, show=True, save=False, Z_comp=None,
              labels=["Data", "Best fit", "Init fit"], append=False, legend=True):
    """Bode plot of impedance.

    Plots phase and log of magnitude over log of frequency.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`, double
        Frequency array
    Z: :class:`numpy.ndarray`, complex
        Impedance array, experimental data or data to compare to.
    Z_fit: :class:`numpy.ndarray`, complex
        Impedance array, fit result. If provided, the difference
        between data and fit will be shown.
    title: str
        Title of plot.
    show: bool, optional
        Show figure (default is True).
    save: bool, optional
        Save figure to pdf (default is False). Name of figure starts with `title`.
    Z_comp: :class:`numpy.ndarray`, complex, optional
        Complex-valued impedance array. Might be used to compare the properties of two data sets.
    labels: list
        List of labels for three plots. Must have length 3 always.
        Is ordered like: `[Z, Z_fit, Z_comp]`
    save: bool, optional
        save figure to pdf (default is False). Name of figure starts with `title`
        and ends with `_bode_plot.pdf`.
    append: bool, optional
        Decide if you want to show plot or add line to existing plot.
    legend: str, optional
        Choose if a legend should be shown. Recommended to switch to False
        when using large datasets.

    """

    axes = []
    plt.figure("bodeimpedance")
    axes = plt.gcf().axes

    if len(axes) < 2:
        plt.suptitle(title, y=1.05)
        plt.subplot(211)
    else:
        plt.sca(axes[0])

    # plot real part of impedance
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r"|Z| / $\Omega$")
    plt.xlabel('Frequency / Hz')
    plt.plot(omega / (2. * np.pi), np.abs(Z), label=labels[0])
    if Z_fit is not None:
        plt.plot(omega / (2. * np.pi), np.abs(Z_fit), '^', label=labels[1])
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), np.abs(Z_comp), 'v', label=labels[2])
    if legend:
        plt.legend()

    if len(axes) < 2:
        plt.subplot(212)
    else:
        plt.sca(axes[1])
    plt.xscale('log')
    plt.ylabel("Phase / °")
    plt.xlabel('Frequency / Hz')
    plt.plot(omega / (2. * np.pi), np.angle(Z, deg=True), label=labels[0])
    if Z_fit is not None:
        plt.plot(omega / (2. * np.pi), np.angle(Z_fit, deg=True), '^', label=labels[1])
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), np.angle(Z_comp, deg=True), 'v', label=labels[2])
    if legend:
        plt.legend()
    plt.tight_layout()
    if save and not append:
        plt.savefig(str(title).replace(" ", "_") + "_bode_plot.pdf")
    if show and not append:
        plt.show()
    elif not show and not append:
        plt.close()


def plot_resistance_capacitance(omega, Z, title="", Z_fit=None, show=True, save=False, Z_comp=None,
                                labels=["Data", "Best fit", "Init fit"], append=False, legend=True):
    """R-C plot of impedance.

    Plots phase and log of magnitude over log of frequency.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`, double
        Frequency array
    Z: :class:`numpy.ndarray`, complex
        Impedance array, experimental data or data to compare to.
    Z_fit: :class:`numpy.ndarray`, complex
        Impedance array, fit result. If provided, the difference
        between data and fit will be shown.
    title: str
        Title of plot.
    show: bool, optional
        Show figure (default is True).
    save: bool, optional
        Save figure to pdf (default is False). Name of figure starts with `title`.
    Z_comp: :class:`numpy.ndarray`, complex, optional
        Complex-valued impedance array. Might be used to compare the properties of two data sets.
    labels: list
        List of labels for three plots. Must have length 3 always.
        Is ordered like: `[Z, Z_fit, Z_comp]`
    save: bool, optional
        save figure to pdf (default is False). Name of figure starts with `title`
        and ends with `_bode_plot.pdf`.
    append: bool, optional
        Decide if you want to show plot or add line to existing plot.
    legend: str, optional
        Choose if a legend should be shown. Recommended to switch to False
        when using large datasets.

    """

    axes = []
    plt.figure("rcimpedance")
    axes = plt.gcf().axes

    if len(axes) < 2:
        plt.suptitle(title, y=1.05)
        plt.subplot(211)
    else:
        plt.sca(axes[0])

    # plot real part of impedance
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r"R / $\Omega$")
    R, C = _return_resistance_capacitance(omega, Z)

    plt.xlabel('Frequency / Hz')
    plt.plot(omega / (2. * np.pi), R, label=labels[0])
    if Z_fit is not None:
        R_fit, C_fit = _return_resistance_capacitance(omega, Z_fit)
        plt.plot(omega / (2. * np.pi), R_fit, '^', label=labels[1])
    if Z_comp is not None:
        R_comp, C_comp = _return_resistance_capacitance(omega, Z_comp)
        plt.plot(omega / (2. * np.pi), C_comp, 'v', label=labels[2])
    if legend:
        plt.legend()

    if len(axes) < 2:
        plt.subplot(212)
    else:
        plt.sca(axes[1])
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("C / F")
    plt.xlabel('Frequency / Hz')
    plt.plot(omega / (2. * np.pi), C, label=labels[0])
    if Z_fit is not None:
        plt.plot(omega / (2. * np.pi), C_fit, '^', label=labels[1])
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), C_comp, 'v', label=labels[2])
    if legend:
        plt.legend()
    plt.tight_layout()
    if save and not append:
        plt.savefig(str(title).replace(" ", "_") + "_rc_plot.pdf")
    if show:
        plt.show()
    elif not show and not append:
        plt.close()


def plot_impedance(omega, Z, title="", Z_fit=None, show=True, save=False, Z_comp=None,
                   labels=["Data", "Best fit", "Init fit"], residual="parts", sign=False,
                   Zlog=False, append=False, limits_residual=None,
                   omega_fit=None, omega_comp=None, legend=True, compare=True):
    """Plot the `result` and compare it to data `Z`.

    Generates 4 subplots showing the real and imaginary parts over
    the frequency; a Nyquist plot of real and negative imaginary part
    and the relative differences of real and imaginary part as well as absolute value of impedance.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`, double
        Frequency array
    Z: :class:`numpy.ndarray`, complex
        Impedance array, experimental data or data to compare to.
    Z_fit: :class:`numpy.ndarray`, complex
        Impedance array, fit result. If provided, the difference
        between data and fit will be shown.
    title: str
        Title of plot.
    show: bool, optional
        Show figure (default is True).
    save: bool, optional
        Save figure to pdf (default is False). Name of figure starts with `title`
        and ends with `_impedance_overview.pdf`.
    Z_comp: :class:`numpy.ndarray`, complex, optional
        Complex-valued impedance array. Might be used to compare the properties of two data sets.
    labels: list
        List of labels for three plots. Must have length 3 always.
        Is ordered like: `[Z, Z_fit, Z_comp]`
    residual: str
        Plot relative difference w.r.t. real and imaginary part if `parts`.
        Plot relative difference w.r.t. absolute value if `absolute`.
        Plot difference (residual) if `diff`.
    sign: bool, optional
        Use sign of residual. Default is False, i.e. absolute value is plotted.
    Zlog: bool, optional
        Log-scale of impedance
    append: bool, optional
        Decide if you want to show plot or add line to existing plot.
    omega_fit: :class:`numpy.ndarray`, double, optional
        Frequency array, provide only if fitted impedance was evaluated at
        different frequencies than the experimental data
    omega_comp: :class:`numpy.ndarray`, double, optional
        Frequency array, provide only if fitted impedance was evaluated at
        different frequencies than the experimental data
    legend: str, optional
        Choose if a legend should be shown. Recommended to switch to False
        when using large datasets.
    compare: bool, optional
        Choose if the difference between fit and data should be computed.


    """

    if omega_fit is None:
        omega_fit = omega
    if omega_comp is None:
        omega_comp = omega
    axes = []
    plt.figure("impedance")
    axes = plt.gcf().axes

    if len(axes) < 3:
        plt.suptitle(title, y=1.05)
        plt.subplot(221)
    else:
        plt.sca(axes[0])
    # use logscale
    if Zlog:
        plt.yscale('log')
    # plot real part of impedance
    plt.xscale('log')
    plt.title("Impedance real part")
    plt.ylabel(r"Re Z / $\Omega$")
    plt.xlabel('Frequency / Hz')
    plt.plot(omega / (2. * np.pi), Z.real, label=labels[0])
    if Z_fit is not None:
        plt.plot(omega_fit / (2. * np.pi), Z_fit.real, linestyle='--', label=labels[1], lw=3)
    if Z_comp is not None:
        plt.plot(omega_comp / (2. * np.pi), Z_comp.real, linestyle='-.', label=labels[2], lw=3)
    if legend:
        plt.legend()
    # plot imaginary part of impedance
    if len(axes) < 3:
        plt.subplot(222)
    else:
        plt.sca(axes[1])
    plt.title("Impedance imaginary part")
    plt.xscale('log')
    plt.ylabel(r"Im Z / $\Omega$")
    plt.xlabel('Frequency / Hz')
    if Zlog:
        plt.yscale('log')
        if np.all(np.less_equal(Z.imag, 0)):
            plt.ylabel(r"-Im Z / $\Omega$")
            plt.plot(omega / (2. * np.pi), -Z.imag, label=labels[0])
            if Z_fit is not None:
                plt.plot(omega_fit / (2. * np.pi), -Z_fit.imag, '--', label=labels[1])
            if Z_comp is not None:
                plt.plot(omega_comp / (2. * np.pi), -Z_comp.imag, '-.', label=labels[2])

        elif np.all(np.greater_equal(Z.imag, 0)):
            plt.plot(omega / (2. * np.pi), Z.imag, label=labels[0])
            if Z_fit is not None:
                plt.plot(omega_fit / (2. * np.pi), Z_fit.imag, '--', label=labels[1])
            if Z_comp is not None:
                plt.plot(omega_comp / (2. * np.pi), Z_comp.imag, '-.', label=labels[2])

        elif np.where(Z.imag < 0)[0].size > np.where(Z.imag > 0)[0].size:
            plt.ylabel(r"-Im Z / $\Omega$")
            plt.plot(omega / (2. * np.pi), -Z.imag, label=labels[0])
            if Z_fit is not None:
                plt.plot(omega_fit / (2. * np.pi), -Z_fit.imag, '--', label=labels[1])
            if Z_comp is not None:
                plt.plot(omega_comp / (2. * np.pi), -Z_comp.imag, '-.', label=labels[2])

        else:
            plt.plot(omega / (2. * np.pi), Z.imag, label=labels[0])
            if Z_fit is not None:
                plt.plot(omega_fit / (2. * np.pi), Z_fit.imag, '--', label=labels[1])
            if Z_comp is not None:
                plt.plot(omega_comp / (2. * np.pi), Z_comp.imag, '-.', label=labels[2])

    else:
        plt.plot(omega / (2. * np.pi), Z.imag, label=labels[0])

        if Z_fit is not None:
            plt.plot(omega_fit / (2. * np.pi), Z_fit.imag, '--', label=labels[1], lw=3)
        if Z_comp is not None:
            plt.plot(omega_comp / (2. * np.pi), Z_comp.imag, '-.', label=labels[2], lw=3)
    if legend:
        plt.legend()
    # plot real vs negative imaginary part
    if len(axes) < 3:
        plt.subplot(223)
    else:
        plt.sca(axes[2])
    plt.title("Nyquist plot")
    plt.ylabel(r"-Im Z / $\Omega$")
    plt.xlabel(r"Re Z / $\Omega$")
    if Zlog:
        plt.xscale('log')
        plt.yscale('log')
    plt.plot(Z.real, -Z.imag, 'o', label=labels[0])
    if Z_fit is not None:
        plt.plot(Z_fit.real, -Z_fit.imag, '^', label=labels[1])
    if Z_comp is not None:
        plt.plot(Z_comp.real, -Z_comp.imag, 'v', label=labels[2])
    if legend:
        plt.legend()
    if Z_fit is not None and np.all(omega == omega_fit) and compare:
        plot_compare_to_data(omega, Z, Z_fit, subplot=224, residual=residual, sign=sign,
                             limits=limits_residual, legend=legend)
    plt.tight_layout()
    if save and not append:
        plt.savefig(str(title).replace(" ", "_") + "_impedance_overview.pdf")
    if show:
        plt.show()
    elif not show and not append:
        plt.close()


def plot_compare_to_data(omega, Z, Z_fit, subplot=None, title="", show=True, save=False,
                         residual="parts", sign=False, limits=None, impedance_threshold=1.,
                         legend=True):
    '''
    plots the difference of the fitted function to the data

    Parameters
    ----------
    omega: :class:`numpy.ndarray`, double
        frequency array
    Z: :class:`numpy.ndarray`, complex
        impedance array, experimental data or data to compare to
    Z_fit: :class:`numpy.ndarray`, complex
        impedance array, fit result
    subplot: optional
        decide whether it is a new figure or a subplot. Default is None (yields new figure). Otherwise it can be an integer to denote the subfigure.
    title: str, optional
        title of plot. Default is an empty string.
    show: bool, optional
        show figure (default is True). Only has an effect when `subplot` is None.
    save: bool, optional
        save figure to pdf (default is False). Name of figure starts with `title`
        and ends with `_relative_difference_to_data.pdf` or `_difference_to_data.pdf`.
    relative: str, optional
        Plot relative difference if True, else plot residual (i.e. just difference).
    sign: bool, optional
        Use sign of residual. Default is False, i.e. absolute value is plotted.
    residual: str
        Plot relative difference w.r.t. real and imaginary part if `parts`.
        Plot relative difference w.r.t. absolute value if `absolute`.
        Plot difference (residual) if `diff`.
    limits: list, optional
        List with entries `[bottom, top]` for y-axis of residual plot.
    impedance_threshold: double, optional
        Threshold for impedance around 0, which is disregarded in the relative
        differences plot. Default is that impedances, with an absolute value less than
        0 are not considered.
    legend: str, optional
        Choose if a legend should be shown. Recommended to switch to False
        when using large datasets.


    Notes
    -----

    When computing the relative difference, impedances between -1 and 1 Ohm are not
    considered since they might lead to a blow up of the relative difference (close to division by 0).
    Instead of this quantitative measure, qualitative checks should be done.
    '''
    if subplot is None:
        plt.figure()
    else:
        show = False
        plt.subplot(subplot)
    if residual == "parts":
        close_to_zero_real = np.where(np.isclose(Z.real, 0., atol=impedance_threshold))
        close_to_zero_imag = np.where(np.isclose(Z.imag, 0., atol=impedance_threshold))
        diff_real = 100. * (Z.real - Z_fit.real) / Z.real
        diff_imag = 100. * (Z.imag - Z_fit.imag) / Z.imag
        diff_real[close_to_zero_real] = np.nan
        diff_imag[close_to_zero_imag] = np.nan
        diff_abs = 100. * np.abs((Z - Z_fit) / Z)
        label = 'Relative difference / %'
    elif residual == "absolute":
        diff_real = 100. * (Z.real - Z_fit.real) / np.abs(Z)
        diff_imag = 100. * (Z.imag - Z_fit.imag) / np.abs(Z)
        label = 'Relative difference / %'
    elif residual == "diff":
        diff_real = Z.real - Z_fit.real
        diff_imag = Z.imag - Z_fit.imag
        diff_abs = np.abs(Z - Z_fit)
        label = r"Difference / $\Omega$"
    else:
        raise RuntimeError("""residual must be either `parts`, `absolute` or `diff`,
                              but not {}""".format(residual))
    if not sign:
        diff_real = np.abs(diff_real)
        diff_imag = np.abs(diff_imag)

    plt.xscale('log')
    if title is not None:
        plt.title((str(title) + "relative difference to data").capitalize())
    plt.xlabel('Frequency / Hz')
    plt.ylabel(label)
    plt.plot(omega / (2. * np.pi), diff_real, label='Real part', lw=3)
    plt.plot(omega / (2. * np.pi), diff_imag, label='Imag part', lw=3, linestyle="--")
    if residual != "absolute":
        plt.plot(omega / (2. * np.pi), diff_abs, label='Abs value', linestyle="-.", lw=3)
    if limits is not None:
        assert len(limits) == 2, "You need to provide upper and lower limit!"
        plt.ylim(limits)
    if legend:
        plt.legend()
    if subplot is None:
        plt.tight_layout()
    if save:
        if residual != "diff":
            plt.savefig(str(title).replace(" ", "_") + "_relative_difference_to_data.pdf")
        else:
            plt.savefig(str(title).replace(" ", "_") + "_difference_to_data.pdf")
    if show:
        plt.show()
    elif not show and subplot is None:
        plt.close()


def emcee_plot(res, clustered=False, **corner_kwargs):
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
    if not clustered:
        flatchain = res.flatchain
    else:
        flatchain = res.new_flatchain
    plot = corner.corner(flatchain, labels=labels,
                         truths=truths, **corner_kwargs)
    return plot


def plot_uncertainty(omega, Zdata, Z, Z1, Z2, sigma, show=True, model=None):
    """Plot best fit with uncertainty interval.

    Parameters
    ----------

    Zdata: :class:`numpy.ndarray`, complex
        impedance array of experimental data
    Z: :class:`numpy.ndarray`, complex
        impedance array of best fit
    Z1: :class:`numpy.ndarray`, complex
        impedance array of upper uncertainty limit
    Z2: :class:`numpy.ndarray`, complex
        impedance array of lower uncertainty limit
    sigma: double
        confidence level
    show: bool, optional
        show figure (default is True)
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
    plt.title("Impedance real part")
    plt.ylabel(r"Re Z / $\Omega$")
    plt.xlabel("Frequency / Hz")
    plt.plot(omega / (2. * np.pi), Z.real, '^', label='Best fit')
    plt.plot(omega / (2. * np.pi), Zdata.real, 'r', label='Data')
    plt.fill_between(omega / (2. * np.pi), Z1.real, Z2.real,
                     color='#888888', label=r"${} \sigma$ interval".format(sigma))
    plt.legend()
    # plot imaginary part of impedance
    plt.subplot(212)
    plt.title("Impedance imaginary part")
    plt.xscale('log')
    plt.ylabel(r"Im Z / $\Omega$")
    plt.xlabel("Frequency / Hz")
    plt.plot(omega / (2. * np.pi), Z.imag, '^', label='Best fit')
    plt.plot(omega / (2. * np.pi), Zdata.imag, 'r', label='Data')
    plt.fill_between(omega / (2. * np.pi), Z1.imag, Z2.imag,
                     color='#888888', label=r"${} \sigma$ interval".format(sigma))
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


def plot_admittance(omega, Z, title="", Z_fit=None, show=True, save=False, Z_comp=None,
                    labels=["Data", "Best fit", "Init fit"], residual="parts", sign=False,
                    Zlog=False, append=False, limits_residual=None,
                    omega_fit=None, omega_comp=None, legend=True, compare=True):
    """Plot the admittance and compare it to data 1/`Z`.

    Generates 4 subplots showing the real and imaginary parts over
    the frequency; a Nyquist plot of real and negative imaginary part
    and the relative differences of real and imaginary part as well as absolute value of admittance.

    Parameters
    ----------
    omega: :class:`numpy.ndarray`, double
        Frequency array
    Z: :class:`numpy.ndarray`, complex
        Impedance array, experimental data or data to compare to.
    Z_fit: :class:`numpy.ndarray`, complex
        Impedance array, fit result. If provided, the difference
        between data and fit will be shown.
    title: str
        Title of plot.
    show: bool, optional
        Show figure (default is True).
    save: bool, optional
        Save figure to pdf (default is False). Name of figure starts with `title`
        and ends with `_admittance_overview.pdf`.
    Z_comp: :class:`numpy.ndarray`, complex, optional
        Complex-valued impedance array. Might be used to compare the properties of two data sets.
    labels: list
        List of labels for three plots. Must have length 3 always.
        Is ordered like: `[Z, Z_fit, Z_comp]`
    residual: str
        Plot relative difference w.r.t. real and imaginary part if `parts`.
        Plot relative difference w.r.t. absolute value if `absolute`.
        Plot difference (residual) if `diff`.
    sign: bool, optional
        Use sign of residual. Default is False, i.e. absolute value is plotted.
    Zlog: bool, optional
        Log-scale of impedance
    append: bool, optional
        Decide if you want to show plot or add line to existing plot.
    omega_fit: :class:`numpy.ndarray`, double, optional
        Frequency array, provide only if fitted impedance was evaluated at
        different frequencies than the experimental data
    omega_comp: :class:`numpy.ndarray`, double, optional
        Frequency array, provide only if fitted impedance was evaluated at
        different frequencies than the experimental data
    legend: str, optional
        Choose if a legend should be shown. Recommended to switch to False
        when using large datasets.
    compare: bool, optional
        Choose if the difference between fit and data should be computed.


    """

    if omega_fit is None:
        omega_fit = omega
    if omega_comp is None:
        omega_comp = omega
    axes = []
    plt.figure("admittance")
    axes = plt.gcf().axes

    if len(axes) < 3:
        plt.suptitle(title, y=1.05)
        plt.subplot(221)
    else:
        plt.sca(axes[0])
    # use logscale
    if Zlog:
        plt.yscale('log')
    # plot real part of admittance
    plt.xscale('log')
    plt.title("Admittance real part")
    plt.ylabel(r"Re Y / S")
    plt.xlabel('Frequency / Hz')
    plt.plot(omega / (2. * np.pi), (1. / Z).real, label=labels[0])
    if Z_fit is not None:
        plt.plot(omega_fit / (2. * np.pi), (1. / Z_fit).real, linestyle='--', label=labels[1], lw=3)
    if Z_comp is not None:
        plt.plot(omega_comp / (2. * np.pi), (1. / Z_comp).real, linestyle='-.', label=labels[2], lw=3)
    if legend:
        plt.legend()
    # plot imaginary part of impedance
    if len(axes) < 3:
        plt.subplot(222)
    else:
        plt.sca(axes[1])
    plt.title("Admittance imaginary part")
    plt.xscale('log')
    plt.ylabel(r"Im Y / S")
    plt.xlabel('Frequency / Hz')
    if Zlog:
        plt.yscale('log')
        if np.all(np.less_equal(Z.imag, 0)):
            plt.ylabel(r"Im Y / S")
            plt.plot(omega / (2. * np.pi), (1. / Z).imag, label=labels[0])
            if Z_fit is not None:
                plt.plot(omega_fit / (2. * np.pi), (1. / Z_fit).imag, '--', label=labels[1])
            if Z_comp is not None:
                plt.plot(omega_comp / (2. * np.pi), (1. / Z_comp).imag, '-.', label=labels[2])

        elif np.all(np.greater_equal(Z.imag, 0)):
            plt.plot(omega / (2. * np.pi), (1. / Z).imag, label=labels[0])
            if Z_fit is not None:
                plt.plot(omega_fit / (2. * np.pi), (1. / Z_fit).imag, '--', label=labels[1])
            if Z_comp is not None:
                plt.plot(omega_comp / (2. * np.pi), (1. / Z_comp).imag, '-.', label=labels[2])

        elif np.where(Z.imag < 0).size > np.where(Z.imag > 0).size:
            plt.ylabel(r"Im Y / S")
            plt.plot(omega / (2. * np.pi), (1. / Z).imag, label=labels[0])
            if Z_fit is not None:
                plt.plot(omega_fit / (2. * np.pi), (1. / Z_fit).imag, '--', label=labels[1])
            if Z_comp is not None:
                plt.plot(omega_comp / (2. * np.pi), (1. / Z_comp).imag, '-.', label=labels[2])

        else:
            plt.plot(omega / (2. * np.pi), (1. / Z).imag, label=labels[0])
            if Z_fit is not None:
                plt.plot(omega_fit / (2. * np.pi), (1. / Z_fit).imag, '--', label=labels[1])
            if Z_comp is not None:
                plt.plot(omega_comp / (2. * np.pi), (1. / Z_comp).imag, '-.', label=labels[2])

    else:
        plt.plot(omega / (2. * np.pi), (1. / Z).imag, label=labels[0])

        if Z_fit is not None:
            plt.plot(omega_fit / (2. * np.pi), (1. / Z_fit).imag, '--', label=labels[1], lw=3)
        if Z_comp is not None:
            plt.plot(omega_comp / (2. * np.pi), (1. / Z_comp).imag, '-.', label=labels[2], lw=3)
    if legend:
        plt.legend()
    # plot real vs negative imaginary part
    if len(axes) < 3:
        plt.subplot(223)
    else:
        plt.sca(axes[2])
    plt.title("Nyquist plot")
    plt.ylabel(r"Im Y / S")
    plt.xlabel(r"Re Y / S")
    if Zlog:
        plt.xscale('log')
        plt.yscale('log')
    plt.plot((1. / Z).real, (1. / Z).imag, 'o', label=labels[0])
    if Z_fit is not None:
        plt.plot((1. / Z_fit).real, (1. / Z_fit).imag, '^', label=labels[1])
    if Z_comp is not None:
        plt.plot((1. / Z_comp).real, (1. / Z_comp).imag, 'v', label=labels[2])
    if legend:
        plt.legend()
    if Z_fit is not None and np.all(omega == omega_fit) and compare:
        plot_compare_to_data(omega, Z, Z_fit, subplot=224, residual=residual, sign=sign,
                             limits=limits_residual, legend=legend)
    plt.tight_layout()
    if save and not append:
        plt.savefig(str(title).replace(" ", "_") + "_admittance_overview.pdf")
    if show:
        plt.show()
    elif not show and not append:
        plt.close()


def plot_dielectric_dispersion(omega, Z, c0, Z_comp=None, title="", show=True, save=False, logscale="permittivity",
                               labels=None, **plotkwargs):
    '''
    Parameters
    ----------

    omega: :class:`numpy.ndarray`, double
        frequency array
    Z: :class:`numpy.ndarray`, complex
        impedance array
    c0: double
        unit capacitance of device
    Z_comp: :class:`numpy.ndarray`, complex, optional
        complex-valued impedance array. Might be used to compare the properties of two data sets.
    title: str, optional
        title of plot. Default is an empty string.
    show: bool, optional
        show figure (default is True)
    save: bool, optional
        save figure to pdf (default is False). Name of figure starts with `title`
        and ends with `_dielectric_properties.pdf`.
    logscale: str, optional
        Decide what you want to plot using log scale.
        Possible are `permittivity`, `conductivity` and `both`
    labels: list, optional
        Give custom labels. Needs to be a list of length 2.
    '''
    eps_r, cond_fit = return_diel_properties(omega, Z, c0)

    fig, ax1 = plt.subplots()

    plt.title(title, y=1.05)

    if labels is None:
        labels = [r'$Z_1$', r'$Z_2$']
    assert len(labels) == 2, "You need to provide lables as a list containing 2 strings!"
    if Z_comp is not None:
        eps_r2, cond_fit2 = return_diel_properties(omega, Z_comp, c0)
    ax1.set_ylabel(r"Relative permittivity")
    ax1.set_xlabel('Frequency / Hz')
    if logscale == 'permittivity' or logscale == 'both':
        ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.plot(omega / (2. * np.pi), eps_r, label=labels[0], **plotkwargs)
    if Z_comp is not None:
        ax1.plot(omega / (2. * np.pi), eps_r2, label=labels[1], **plotkwargs)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Dielectric loss")
    if logscale == 'conductivity' or logscale == 'both':
        ax2.set_yscale('log')
    ax2.plot(omega / (2. * np.pi), cond_fit / (e0 * omega), ls="-.", **plotkwargs)
    if Z_comp is not None:
        plt.plot(omega / (2. * np.pi), cond_fit2 / (e0 * omega), ls="-.", **plotkwargs)
    ax1.legend()
    fig.tight_layout()
    if save:
        plt.savefig(str(title).replace(" ", "_") + "_dielectric_dispersion.pdf")
    if show:
        plt.show()
    else:
        plt.close()
