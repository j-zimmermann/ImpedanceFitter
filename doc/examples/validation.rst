Validating the data
-------------------

Before fitting the experimental data to an equivalent
circuit model, you can make sure that the data are valid.
The LinKK test [Schoenleber2014]_ allows one to validate
the data.

Here, it is summarized how the test works in ImpedanceFitter.

Assume the model discussed in [Schoenleber2014]_.
It can be represented in ImpedanceFitter by

.. code-block:: python

	model = 'R_s1 + parallel(C_s3, R_s2)  + parallel(R_s4, W_s5)'	

After having generated artifical data stored in `test.csv`,
we can initialize the fitter.

.. code-block:: python

	fitter = impedancefitter.Fitter('CSV')

Then, the LinKK test can be performed for all data sets by running

.. code-block:: python

	results, mus, residuals = fitter.linkk_test()

`results` contains the fit results as a dictionary.
`mus` is a dictionary with all :math:`\mu` values for an increasing number of RC-elements used in the LinKK-test. 
`residuals` is a dictionary containing all residuals during the least-squares fit.

The result of the fit looks like this:

.. image:: impedance-linkk.*
        :width: 600

The parameter :math:`\mu` decays with an increased number of RC elements
as described in [Schoenleber2014]_.
It is used to detect overfitting. The threshold for overfitting is set to
`0.85` but can be manually adjusted.

.. image:: mu-linkk.*
        :width: 600

In this example, all residuals are very small (as expected for artifical data).
If the relative difference exceeds 1% or if there is a drift
in the residuals, concerns about the validity of
the experimental data could be raised.  
If you observe sinusoidal oscillations in your residuals,
increase the number of RC-elements either manually or by decreasing
the threshold to values below `0.85`.

If there is an inductive or capacitive element present, it can be benefitial
to add an extra capacitance or inductance to the circuit.
This can be done by 

.. code-block:: python

	results, mus = fitter.linkk_test(capacitance=True)
	results, mus = fitter.linkk_test(inductance=True)
	results, mus = fitter.linkk_test(capacitance=True, inductance=True)

Especially if you observe large residuals at high frequencies, an inductive element
should be added.

See Also
^^^^^^^^

:download:`examples/LinKK/linkk.py <../../examples/LinKK/linkk.py>`.
:download:`examples/LinKK/linkk_cap.py <../../examples/LinKK/linkk_cap.py>`.
:download:`examples/LinKK/linkk_ind.py <../../examples/LinKK/linkk_ind.py>`.
:download:`examples/LinKK/linkk_ind_cap.py <../../examples/LinKK/linkk_ind_cap.py>`.

:meth:`impedancefitter.fitter.Fitter.linkk_test`
