.. ImpedanceFitter documentation master file, created by
   sphinx-quickstart on Thu Aug 29 12:24:51 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ImpedanceFitter's documentation!
===========================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   examples
   fitting
   statistics
   elements
   circuits
   utils
   plotting

.. _Overview:

Overview
========

Impedance spectroscopy (IS) is a great tool to analyse the behaviour of an electrical circuit,
to characterise the response of a sample (e.g. biological tissue), to determine the dielectric 
properties of a sample, and much more [Barsoukov2018]_.

In IS, often (complex) non-linear least squares is used for parameter estimation
of equivalent circuit models.
ImpedanceFitter is a software that facilitates parameter estimation for arbitrary equivalent circuit models.
The equivalent circuit may comprise different standard elements or 
other models that have been formulated in the context of impedance spectroscopy.
The unknown parameters are found by fitting the model to experimental impedance data.
The underlying fitting software is LMFIT [Newville2019]_, which offers an interface to different optimization and curve-fitting 
methods going beyond standard least-squares.
ImpedanceFitter allows one to build a custom equivalent circuit, fit an arbitrary amount of data sets and 
perform statistical analysis of the results using OpenTurns [Baudin2017]_.

.. [Newville2019]  Newville, M., Otten, R., Nelson, A., Ingargiola, A., Stensitzki, T., Allan, D., Fox, A., Carter, F., Michał, Pustakhod, D., Ram, Y., Glenn, Deil, C., Stuermer, Beelen, A., Frost, O., Zobrist, N., Pasquevich, G., Hansen, A.L.R., Spillane, T., Caldwell, S., Polloreno, A., andrewhannum, Zimmermann, J., Borreguero, J., Fraine, J., deep-42-thought, Maier, B.F., Gamari, B., & Almarza, A.
                   (2019, December 20). lmfit/lmfit-py 1.0.0 (Version 1.0.0). Zenodo. http://doi.org/10.5281/zenodo.3588521

.. [Baudin2017] Baudin, M., Dutfoy, A., Looss, B., & Popelin, A. L. (2017). 
                OpenTURNS: An industrial software for uncertainty quantification in simulation.
                In Handbook of Uncertainty Quantification (pp. 2001–2038). https://doi.org/10.1007/978-3-319-12385-1_64

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
