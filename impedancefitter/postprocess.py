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
from math import ceil
import openturns as ot
from openturns.viewer import View
import yaml
from .utils import get_labels
import logging

logger = logging.getLogger(__name__)


class PostProcess(object):
    """This class provides the possibility to statistically analyse the fitted data.

    Parameters
    ----------

    fitresult: dict
        Result of the fit.
    yamlfile: bool
        Provide the link to a file from which you want to
        read the results.

    Notes
    -----

    Provide either `fitresult` or `yamlfile`.

    """
    def __init__(self, fitresult=None, yamlfile=False):

        if fitresult is not None and yamlfile is False:
            self.data = fitresult
        elif yamlfile is not None and fitresult is None:
            with open(yamlfile, 'r') as fitfile:
                self.data = yaml.safe_load(fitfile)
        else:
            raise RuntimeError("Provide either yamlfile or fitresult.")
        assert isinstance(self.data, dict),\
            "The fit result to be analysed needs to be a dictionary."

        random_key = next(iter(self.data))
        self.parameters = list(self.data[random_key].keys())
        self.labels = get_labels(self.parameters)
        # write data into dict
        self.sampledict = {}
        for p in self.parameters:
            plist = []
            for values in self.data.values():
                try:
                    plist.append([values[p]])
                except KeyError:
                    print("""There must be all parameters present
                             over the entire data set you want to analyse.""")
            if np.all(np.isclose(plist, plist[0])):
                logger.info("All values for parameter {} are equal. Parameter will be neglected since it was kept constant.".format(p))
                continue
            self.sampledict[p] = ot.Sample(np.array(plist))
        self.parameters = list(self.sampledict.keys())

    def plot_histograms(self, savefig=False, show=True):
        """Plot histograms for all determined parameters.

        Parameters
        ----------

        savefig: bool, optional
            Set to True if you want to save the figure `histograms.pdf`.
        show: bool, optional
            Switch on or off if figures is shown.
        Notes
        -----

        Fails if values are too close to each other, i.e.
        the variance is very small.
        """
        if len(self.parameters) < 3:
            ncols = len(self.parameters)
            nrows = 1
        else:
            ncols = 3
            nrows = int(len(self.parameters) / 3)
            nrows += int(ceil((len(self.parameters) % 3) / 3))
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        r = 0
        c = 0
        for key in self.sampledict:
            graph = ot.HistogramFactory().build(self.sampledict[key]).drawPDF()
            graph.setTitle("Histogram for variables")
            graph.setXTitle(self.labels[key])
            if nrows == 1:
                View(graph, axes=[ax[c]], plot_kwargs={'label': "hist", 'c': 'black'})
                ymin, ymax = ax[c].get_ylim()
            else:
                View(graph, axes=[ax[r, c]], plot_kwargs={'label': "hist", 'c': 'black'})
                ymin, ymax = ax[r, c].get_ylim()
            kernel = ot.KernelSmoothing()
            graph_k = kernel.build(self.sampledict[key])
            graph_k = graph_k.drawPDF()
            graph_k.setTitle("Histogram for variables")
            graph_k.setXTitle(key)
            if nrows == 1:
                View(graph_k, axes=[ax[c]], plot_kwargs={'label': "smooth"})
                ymin1, ymax1 = ax[c].get_ylim()
                if ymax1 < ymax:
                    ax[c].set_ylim(ymin1, ymax)

            else:
                View(graph_k, axes=[ax[r, c]], plot_kwargs={'label': "smooth"})
                ymin1, ymax1 = ax[r, c].get_ylim()
                if ymax1 < ymax:
                    ax[r, c].set_ylim(ymin1, ymax)

            # jump to next ax object or next row
            c += 1
            if(c == 3):
                c = 0
                r += 1
        plt.tight_layout()
        if savefig:
            plt.savefig("histograms.pdf")
        if show:
            plt.show()

    def fit_to_normal_distribution(self, parameter, showQQ=False):
        """Fit results for to normal distribution.

        Parameters
        ----------
        parameter: string
            Parameter, whose distribution is to be found.
        showQQ: bool, optional
            Decide if you want to check the fit visually

        Returns
        -------
        :class:`openturns.Distribution`
        """
        sample = self.sampledict[parameter]
        distribution = ot.NormalFactory().build(sample)
        logger.debug(distribution)
        if showQQ:
            # Draw QQ plot to check fitted distribution
            QQ_plot = ot.VisualTest.DrawQQplot(sample, distribution)
            View(QQ_plot).show()
        return distribution

    def fit_to_histogram_distribution(self, parameter, showQQ=False):
        """Generate histogram from results.

        Parameters
        ----------
        parameter: string
            Parameter, whose distribution is to be found.

        Returns
        -------
        :class:`openturns.Distribution`
        """
        sample = self.sampledict[parameter]
        distribution = ot.HistogramFactory().build(sample)
        logger.debug(distribution)
        if showQQ:
            # Draw QQ plot to check fitted distribution
            QQ_plot = ot.VisualTest.DrawQQplot(sample, distribution)
            View(QQ_plot).show()
        return distribution

    def best_model_kolmogorov(self, parameter, distributions, showQQ=False):
        """Test, which distribution models your data best based on the kolmogorov test.

        Parameters
        ----------

        parameter: string
            Parameter, whose distribution is to be found.
        distributions: list
            List with strings describing valid OpenTURNS distributions
            such as `['Normal', 'Uniform']`

        Returns
        -------
            :class:`openturns.Distribution`
            :class:`openturns.TestResult`

        See Also
        --------
        :func:`openturns.FittingTest_BestModelKolmogorov`

        """
        sample = self.sampledict[parameter]
        tested_distributions = []
        for dist in distributions:
            tested_distributions.append(eval("ot." + dist + "Factory()"))
        best_model, best_result = ot.FittingTest.BestModelKolmogorov(sample, tested_distributions)
        logger.debug("Best model:")
        logger.debug(best_model)
        logger.debug("P-value:")
        logger.debug(best_result.getPValue())
        if showQQ:
            logger.debug("QQ Plot for best model:")
            QQ_plot = ot.VisualTest.DrawQQplot(sample, best_model)
            View(QQ_plot).show()
        return best_model, best_result

    def best_model_bic(self, parameter, distributions, showQQ=False):
        """
        Test, which distribution models your data best based on the Bayesian information criterion.

        Parameters
        ----------

        parameter: string
            Parameter, whose distribution is to be found.
        distributions: list
            List with strings describing valid OpenTURNS distributions
            such as `['Normal', 'Uniform']`

        See Also
        --------
            :func:`openturns.FittingTest_BestModelBIC

        Returns
        -------
            :class:`openturns.Distribution`
            float
        """
        ot.RandomGenerator.SetSeed(0)
        sample = self.sampledict[parameter]
        tested_distributions = []
        for dist in distributions:
            tested_distributions.append(eval("ot." + dist + "Factory()"))
        best_model, best_result = ot.FittingTest.BestModelBIC(sample, tested_distributions)
        logger.debug("Best model:")
        logger.debug(best_model)
        logger.debug("Bayesian information criterion:")
        logger.debug(best_result)
        if showQQ:
            logger.debug("QQ Plot for best model:")
            QQ_plot = ot.VisualTest.DrawQQplot(sample, best_model)
            View(QQ_plot).show()
        return best_model, best_result

    def best_model_chisquared(self, parameter, distributions, showQQ=False):
        """Test, which distribution models your data best based on the chisquared test.

        Parameters
        ----------

        parameter: string
            Parameter, whose distribution is to be found.
        distributions: list
            List with strings describing valid OpenTURNS distributions
            such as `['Normal', 'Uniform']`

        See Also
        --------
            :func:`openturns.FittingTest_BestModelChiSquared`

        Returns
        -------
            :class:`openturns.Distribution`
            :class:`openturns.TestResult`
        """

        sample = self.sampledict[parameter]
        tested_distributions = []
        for dist in distributions:
            tested_distributions.append(eval("ot." + dist + "Factory()"))
        best_model, best_result = ot.FittingTest.BestModelChiSquared(sample, tested_distributions)
        logger.debug("Best model:")
        logger.debug(best_model)
        logger.debug("P-value:")
        logger.debug(best_result.getPValue())
        if showQQ:
            logger.debug("QQ Plot for best model:")
            QQ_plot = ot.VisualTest.DrawQQplot(sample, best_model)
            View(QQ_plot).show()
        return best_model, best_result
