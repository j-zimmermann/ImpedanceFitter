Overview
========

Background
----------

Impedance spectroscopy is a great tool to analyse the behaviour of an electrical circuit,
characterise the response of a sample (e.g. biological tissue) or determine the dielectric 
properties of a sample.
ImpedanceFitter is a software that facilitates parameter estimation for various mechanistic models.
The mechanistic model is based on an equivalent circuit that may comprise different standard elements or 
other models that have been formulated in the context of impedance spectroscopy.
The unknown parameters are found by fitting to experimental data.
The underlying fitting software is [LMFIT]_, which offers an interface to different optimization and curve-fitting 
methods.
ImpedanceFitter allows one to build a custom equivalent circuit, fit an arbitrary amount of data sets and 
perform statistical analysis of the results.

.. [LMFIT]  Matt Newville, Renee Otten, Andrew Nelson, Antonino Ingargiola, Till Stensitzki, Dan Allan, Austin Fox, Faustin Carter, Michał, Dima Pustakhod, Yoav Ram, Glenn, Christoph Deil, Stuermer, Alexandre Beelen, Oliver Frost, Nicholas Zobrist, Gustavo Pasquevich, Allan L. R. Hansen, Tim Spillane, Shane Caldwell, Anthony Polloreno, andrewhannum, Julius Zimmermann, Jose Borreguero, Jonathan Fraine, deep-42-thought, Benjamin F. Maier, Ben Gamari, Anthony Almarza. (2019, December 20). lmfit/lmfit-py 1.0.0 (Version 1.0.0). Zenodo. http://doi.org/10.5281/zenodo.3588521

How it works
============

Fitting
-------

The script will cycle through all files in a selected directory
(unless certain files are excluded or explicitly listed)
and will store the experimental data.

Formulate the model
^^^^^^^^^^^^^^^^^^^

The ImpedanceFitter parser understands circuits that follow
a simple pattern:

- Elements in series are connected by a `+`.
- Elements in parallel are connected by `parallel(A, B)`.

An example of a circuit could be:

.. code-block:: python

        parallel(R, C) + CPE

This stands for a resistor in parallel with a capacitor that are in series
with a constant phase element (CPE).

Also nested parallels are possible:

.. code-block:: python

        parallel(parallel(L, C), R)



Execute the fit
^^^^^^^^^^^^^^^

Using :meth:`impedancefitter.main.Fitter.run`, those files can be fitted 
to an equivalent circuit model. 
If there are two models involved that shall be fitted sequentially for each file,
refer to :meth:`impedancefitter.main.Fitter.sequential_run`.
This method allows one to communicate inferered parameters to the second model.
In [Sab2012]_, an example of such a sequential procedure has been presented.

.. [Sab2012] Sabuncu, A. C., Zhuang, J., Kolb, J. F., & Beskok, A. (2012). Microfluidic impedance spectroscopy as a tool for quantitative biology and biotechnology. Biomicrofluidics, 6(3). https://doi.org/10.1063/1.4737121

API Reference
^^^^^^^^^^^^^

.. automodule:: impedancefitter.main
        :members:

Statistical analysis
--------------------

