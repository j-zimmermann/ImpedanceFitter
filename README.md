ImpedanceFitter
===============

Impedance spectroscopy (IS) is a great tool to analyse the behaviour of an electrical circuit,
to characterise the response of a sample (e.g. biological tissue), to determine the dielectric 
properties of a sample, and much more [^Barsoukov2018].

[^Barsoukov2018]: Barsoukov, E., & Macdonald, J. R. (Eds.). (2018). Impedance Spectroscopy: Theory, Experiment, and Applications. (3rd ed.). Hoboken, NJ: John Wiley & Sons, Inc. https://doi.org/10.1002/9781119381860

In IS, often (complex) non-linear least squares is used for parameter estimation
of equivalent circuit models.
ImpedanceFitter is a software that facilitates parameter estimation for arbitrary equivalent circuit models.
The equivalent circuit may comprise different standard elements or 
other models that have been formulated in the context of impedance spectroscopy.
The unknown parameters are found by fitting the model to experimental impedance data.
The underlying fitting software is LMFIT [^N2019], 
which offers an interface to different optimization and curve-fitting 
methods going beyond standard least-squares.

[^N2019]: Newville, M., & et al. (2020, May 7). lmfit/lmfit-py 1.0.1 (Version 1.0.1). Zenodo. http://doi.org/10.5281/zenodo.3814709

ImpedanceFitter allows one to build a custom equivalent circuit, fit an arbitrary amount of data sets and 
perform statistical analysis of the results using OpenTurns [^Baudin2017].

[^Baudin2017]: Baudin, M., Dutfoy, A., Looss, B., & Popelin, A. L. (2017). OpenTURNS: An industrial software for uncertainty quantification in simulation. In Handbook of Uncertainty Quantification (pp. 2001–2038). https://doi.org/10.1007/978-3-319-12385-1_64

Documentation
-------------

The documentation is available at [INSERT READTHEDOCS LINK].

If you want to compile it locally:
The documentation is in the `doc`
directory and requires [Sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html)
to be compiled.


Installation
------------

ImpedanceFitter works with Python >= 3.6.

ImpedanceFitter can be installed using pip
(not yet working, WIP)

```
pip install impedancefitter
```

If you want to install the code from source,
clone into a local directory.

`cd` into this directory and run

```
pip install -e . --user
```

in this directory.
It will install all requirements automatically.
Moreover, you can edit the source code and
run the edited version without reinstalling.

Testing
-------

The tests use [pytest](https://docs.pytest.org/en/latest/).
Simply run `pytest` inside the repository main directory.

Use ImpedanceFitter
-------------------

Check out the `examples` directory and the documentation to see how 
ImpedanceFitter is supposed to work.


Contribute
----------

If you find bugs or missing functionality,
feel free to raise an issue here on github
or create a pull request!

References
----------