Fitting experimental data
-------------------------

Impedance spectroscopy data can be processed
using :class:`impedancefitter.main.Fitter`.

Currently, a few fileformats can be used.
They are summarized in :func:`impedancefitter.utils.available_file_format`.

In this example, artificial data will be generated and fitted.

.. jupyter-execute:: ../../examples/Randles/randles_data.py
        :stderr:

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

