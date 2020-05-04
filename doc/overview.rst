.. _Overview:

Overview
========

Impedance spectroscopy (IS) is a great tool to analyse the behaviour of an electrical circuit,
characterise the response of a sample (e.g. biological tissue) or determine the dielectric 
properties of a sample [1]_.

Data analysis in IS relies often on non-linear least squares for parameter estimation
of equivalent circuit models.
ImpedanceFitter is a software that facilitates parameter estimation for various mechanistic models.
The mechanistic model is based on an equivalent circuit that may comprise different standard elements or 
other models that have been formulated in the context of impedance spectroscopy.
The unknown parameters are found by fitting to experimental impedance data.
The underlying fitting software is LMFIT [2]_, which offers an interface to different optimization and curve-fitting 
methods.
ImpedanceFitter allows one to build a custom equivalent circuit, fit an arbitrary amount of data sets and 
perform statistical analysis of the results.

.. [1] Barsoukov, E., & Macdonald, J. R. (Eds.). (2018). Impedance Spectroscopy: Theory, Experiment, and Applications. (3rd ed.). Hoboken, NJ: John Wiley & Sons, Inc. https://doi.org/10.1002/9781119381860

.. [2]  Matt Newville, Renee Otten, Andrew Nelson, Antonino Ingargiola, Till Stensitzki, Dan Allan, Austin Fox, Faustin Carter, Michał, Dima Pustakhod, Yoav Ram, Glenn, Christoph Deil, Stuermer, Alexandre Beelen, Oliver Frost, Nicholas Zobrist, Gustavo Pasquevich, Allan L. R. Hansen, Tim Spillane, Shane Caldwell, Anthony Polloreno, andrewhannum, Julius Zimmermann, Jose Borreguero, Jonathan Fraine, deep-42-thought, Benjamin F. Maier, Ben Gamari, Anthony Almarza. (2019, December 20). lmfit/lmfit-py 1.0.0 (Version 1.0.0). Zenodo. http://doi.org/10.5281/zenodo.3588521
