.. _Fitting:

Fitting
=======

The script will cycle through all files in a selected directory
(unless certain files are excluded or explicitly listed)
and will store the experimental data.
The experimental data can then be fitted to user-defined model.

Formulate the model
-------------------

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

Find all available elements in Section :ref:`elements`
and all available circuits in Section :ref:`circuits`.

You can also use prefixes. This is needed if you want
to combine multiple elements or circuits of the same type.
Otherwise, the parameters cannot be distinguished by LMFIT.

For example:

.. code-block:: python

        parallel(R_f1, C_f1) + parallel(R_f2, C_f2)

Execute the fit
---------------

Using :class:`impedancefitter.Fitter.run`, those files can be fitted 
to an equivalent circuit model. 
If there are two models involved that shall be fitted sequentially for each file,
refer to :meth:`impedancefitter.Fitter.sequential_run`.
This method allows one to communicate inferered parameters to the second model.
In [3]_, an example of such a sequential procedure has been presented.

.. [3] Sabuncu, A. C., Zhuang, J., Kolb, J. F., & Beskok, A. (2012). Microfluidic impedance spectroscopy as a tool for quantitative biology and biotechnology. Biomicrofluidics, 6(3). https://doi.org/10.1063/1.4737121

API Reference
-------------

.. autoclass:: impedancefitter.main.Fitter
        :members: