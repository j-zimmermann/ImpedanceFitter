.. _example_fitting:

Fitting experimental data
-------------------------

Impedance spectroscopy data can be processed
using :class:`impedancefitter.fitter.Fitter`.

Currently, a few fileformats can be used.
They are summarized in :func:`impedancefitter.utils.available_file_format`.

In this example, artificial data will be generated and fitted.

.. jupyter-execute::
        :hide-code:

	import numpy
	import os
	import impedancefitter
	import pandas

	from matplotlib import rcParams
	rcParams['figure.figsize'] = [15, 10]


We want to fit a user-defined circuit, which reads

.. jupyter-execute::

	model = 'R_s + parallel(R_ct + W, C)'


First the data is generated using the following parameters

.. jupyter-execute::
        :stderr:

	frequencies = numpy.logspace(0, 8)
	Rct = 100.
	Rs = 20.
	Aw = 300.
	C0 = 25e-6


Then the model is defined and data generated and exported

.. jupyter-execute::

	lmfit_model = impedancefitter.get_equivalent_circuit_model(model)
	Z = lmfit_model.eval(omega=2. * numpy.pi * frequencies,
			     ct_R=Rct, s_R=Rs,
			     C=C0, Aw=Aw)
	data = {'freq': frequencies, 'real': Z.real,
		'imag': Z.imag}
	# write data to csv file
	df = pandas.DataFrame(data=data)
	df.to_csv('test.csv', index=False)

The fitter is initialized with verbose output.
Also, the fit results will be plotted immediately.

.. jupyter-execute::

	fitter = impedancefitter.Fitter('CSV', LogLevel='DEBUG', show=True)
	os.remove('test.csv')

We use the Randles circuit that corresponds to the custom circuit model.
The initial guess is passed in a dictionary with minimal information.


.. jupyter-execute::

	model = 'Randles'
	parameters = {'Rct': {'value': 3. * Rct},
		      'Rs': {'value': 0.5 * Rs},
		      'C0': {'value': 0.1 * C0},
		      'Aw': {'value': 1.2 * Aw}}

Then the fit is simply run by 


.. jupyter-execute::
	:stderr:

	fitter.run(model, parameters=parameters)


Initial and best fit can be plotted together to improve on the 
initial parameter guess

.. jupyter-execute::

	fitter.plot_initial_best_fit()


Here, the initial guess was passed in a dictionary with minimal information.
One could also specify bounds or fix a parameter.

For example, if `Rct` was restricted to be between 50 and 500
and `C0` was known and thus fixed, the parameters would read

.. code-block:: python
        
        parameters = {'Rct': {'value': 3. * Rct,
                              'min': 50,
                              'max': 500},
                      'Rs': {'value': 0.5 * Rs},
                      'C0': {'value': C0, 'vary': False},
                      'Aw': {'value': 1.2 * Aw}}

                      
See Also
^^^^^^^^

:download:`examples/Randles/randles_data.py <../../examples/Randles/randles_data.py>`.

