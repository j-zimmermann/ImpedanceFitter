.. _Statistical Analysis:

Statistical analysis
====================

To analyse the fit results
when the data set is rather large,
there exists an interface to `OpenTurns <http://openturns.org>`_.
It can generate histograms or find the best distribution
to describe a certain fit parameter of the data set.

.. note::

   OpenTurns is an optional dependency.
   To use the statistical analysis, install it via

   .. code-block:: bash

      pip install impedancefitter[stats]

API Reference
-------------

.. autoclass:: impedancefitter.postprocess.PostProcess
        :members:

