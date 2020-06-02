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

Data analysis in IS relies often on non-linear least squares for parameter estimation
of equivalent circuit models.
ImpedanceFitter is a software that facilitates parameter estimation for arbitrary equivalent circuit models.
The equivalent circuit may comprise different standard elements or 
other models that have been formulated in the context of impedance spectroscopy.
The unknown parameters are found by fitting the model to experimental impedance data.
The underlying fitting software is LMFIT [Newville2019]_, which offers an interface to different optimization and curve-fitting 
methods going beyond standard least-squares.
ImpedanceFitter allows one to build a custom equivalent circuit, fit an arbitrary amount of data sets and 
perform statistical analysis of the results.

.. [Barsoukov2018] Barsoukov, E., & Macdonald, J. R. (Eds.). (2018). Impedance Spectroscopy: Theory, Experiment, and Applications. (3rd ed.). Hoboken, NJ: John Wiley & Sons, Inc. https://doi.org/10.1002/9781119381860

.. [Newville2019]  Matt Newville, Renee Otten, Andrew Nelson, Antonino Ingargiola, Till Stensitzki, Dan Allan, Austin Fox, Faustin Carter, Michał, Dima Pustakhod, Yoav Ram, Glenn, Christoph Deil, Stuermer, Alexandre Beelen, Oliver Frost, Nicholas Zobrist, Gustavo Pasquevich, Allan L. R. Hansen, Tim Spillane, Shane Caldwell, Anthony Polloreno, andrewhannum, Julius Zimmermann, Jose Borreguero, Jonathan Fraine, deep-42-thought, Benjamin F. Maier, Ben Gamari, Anthony Almarza. (2019, December 20). lmfit/lmfit-py 1.0.0 (Version 1.0.0). Zenodo. http://doi.org/10.5281/zenodo.3588521


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
