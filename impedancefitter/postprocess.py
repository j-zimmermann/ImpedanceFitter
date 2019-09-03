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
import openturns as ot
from openturns.viewer import View
import yaml
import logging

logger = logging.getLogger('impedancefitter-logger')


class PostProcess(object):
    """
    This class provides the possibility, to analyse the statistics of the fitted data.
    The fitting results are read in from the outfile.

    Parameters
    ----------

    model: {string, DoubleShell OR SingleShell}
         define, which model has been used
    electrode_polarization: {bool, default True}
         set to false if not used
    yamlfile: {string, default None}
         define path to yamlfile or use current working directory
    """
    def __init__(self, model, electrode_polarization=True, yamlfile=None):
        self.model = model
        self.electrode_polarization = electrode_polarization
        if yamlfile is None:
            yamlfile = 'outfile.yaml'
        file = open(yamlfile, 'r')
        data = yaml.safe_load(file)
        alphalist, emlist, klist, kmlist, kcplist, kmedlist, emedlist, enelist, knelist, knplist = ([] for i in range(10))
        for key in data:
            if self.electrode_polarization is True:
                alphalist.append([data[key]['alpha']])
                klist.append([data[key]['k']])
            emlist.append([data[key]['em']])
            kmlist.append([data[key]['km']])
            kcplist.append([data[key]['kcp']])
            kmedlist.append([data[key]['kmed']])
            emedlist.append([data[key]['emed']])
            if(self.model == 'DoubleShell'):
                enelist.append([data[key]['ene']])
                knelist.append([data[key]['kne']])
                knplist.append([data[key]['knp']])
        # write data into dict
        self.sampledict = {}
        if self.electrode_polarization is True:
            self.sampledict['alpha'] = ot.Sample(np.array(alphalist))
            self.sampledict['k'] = ot.Sample(np.array(klist))
        self.sampledict['em'] = ot.Sample(np.array(emlist))
        self.sampledict['km'] = ot.Sample(np.array(kmlist))
        self.sampledict['kcp'] = ot.Sample(np.array(kcplist))
        self.sampledict['kmed'] = ot.Sample(np.array(kmedlist))
        self.sampledict['emed'] = ot.Sample(np.array(emedlist))
        if(self.model == 'DoubleShell'):
            self.sampledict['ene'] = ot.Sample(np.array(enelist))
            self.sampledict['kne'] = ot.Sample(np.array(knelist))
            self.sampledict['knp'] = ot.Sample(np.array(knplist))

    def plot_histograms(self):
        """
        Plot histograms for all determined parameters.
        fails if values are too close to each other
        """
        nrows = 3
        ncols = 3
        if self.electrode_polarization is not True:
            nrows -= 1
        if(self.model == 'DoubleShell'):
            nrows += 1
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        r = 0
        c = 0
        for key in self.sampledict:
            graph = ot.HistogramFactory().build(self.sampledict[key]).drawPDF()
            graph.setTitle("Histogram for variables")
            graph.setXTitle(key)
            View(graph, axes=[ax[r, c]], plot_kwargs={'label': key, 'c': 'black'})
            kernel = ot.KernelSmoothing()
            graph_k = kernel.build(self.sampledict[key])
            graph_k = graph_k.drawPDF()
            graph_k.setTitle("Histogram for variables")
            graph_k.setXTitle(key)
            View(graph_k, axes=[ax[r, c]], plot_kwargs={'label': key})
            # jump to next ax object or next row
            c += 1
            if(c == 3):
                c = 0
                r += 1
        plt.tight_layout()
        plt.show()

    def fit_to_normal_distribution(self, parameter):
        """
        Fit results for to normal distribution.

        :param parameter: string, parameter that is to be fitted.
        """
        sample = self.sampledict[parameter]
        distribution = ot.NormalFactory().build(sample)
        print(distribution)
        # Draw QQ plot to check fitted distribution
        QQ_plot = ot.VisualTest.DrawQQplot(sample, distribution)
        View(QQ_plot).show()
        return distribution

    def fit_to_histogram_distribution(self, parameter):
        """
        Generate histogram from results.

        :param parameter: string, parameter that is to be fitted.
        """
        sample = self.sampledict[parameter]
        distribution = ot.HistogramFactory().build(sample)
        print(distribution)
        # Draw QQ plot to check fitted distribution
        QQ_plot = ot.VisualTest.DrawQQplot(sample, distribution)
        View(QQ_plot).show()
        return distribution

    def best_model_kolmogorov(self, parameter, distributions):
        """
        Test, which distribution models your data best based on the kolmogorov test (see https://openturns.github.io/openturns/master/user_manual/_generated/openturns.FittingTest_BestModelKolmogorov.html#openturns.FittingTest_BestModelKolmogorov).
        suitable for small samples

        :param distributions: list of strings like: :code:`['Normal', 'Uniform']`
        """
        sample = self.sampledict[parameter]
        tested_distributions = []
        for dist in distributions:
            tested_distributions.append(eval("ot." + dist + "Factory()"))
        best_model, best_result = ot.FittingTest.BestModelKolmogorov(sample, tested_distributions)
        logger.info("Best model:")
        logger.info(best_model)
        logger.info("P-value:")
        logger.info(best_result.getPValue())
        logger.info("QQ Plot for best model:")
        QQ_plot = ot.VisualTest.DrawQQplot(sample, best_model)
        View(QQ_plot).show()
        return best_model

    def best_model_bic(self, parameter, distributions):
        """
        Test, which distribution models your data best based on the Bayesian information criterion (see https://openturns.github.io/openturns/master/user_manual/_generated/openturns.FittingTest_BestModelBIC.html#openturns.FittingTest_BestModelBIC).
        suitable for small samples

        :param distributions: list of strings like: :code:`['Normal', 'Uniform']`
        """
        ot.RandomGenerator.SetSeed(0)
        sample = self.sampledict[parameter]
        tested_distributions = []
        for dist in distributions:
            tested_distributions.append(eval("ot." + dist + "Factory()"))
        best_model, best_result = ot.FittingTest.BestModelBIC(sample, tested_distributions)
        logger.info("Best model:")
        logger.info(best_model)
        logger.info("Bayesian information criterion:")
        logger.info(best_result)
        logger.info("QQ Plot for best model:")
        QQ_plot = ot.VisualTest.DrawQQplot(sample, best_model)
        View(QQ_plot).show()
        return best_model

    def best_model_chisquared(self, parameter, distributions):
        """
        Test, which distribution models your data best based on the chisquared test (see https://openturns.github.io/openturns/master/user_manual/_generated/openturns.FittingTest_BestModelChiSquared.html).

        :param distributions: list of strings like: :code:`['Normal', 'Uniform']`
        """
        sample = self.sampledict[parameter]
        tested_distributions = []
        for dist in distributions:
            tested_distributions.append(eval("ot." + dist + "Factory()"))
        best_model, best_result = ot.FittingTest.BestModelChiSquared(sample, tested_distributions)
        logger.info("Best model:")
        logger.info(best_model)
        logger.info("P-value:")
        logger.info(best_result.getPValue())
        logger.info("QQ Plot for best model:")
        QQ_plot = ot.VisualTest.DrawQQplot(sample, best_model)
        View(QQ_plot).show()
        return best_model
