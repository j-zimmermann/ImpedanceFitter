Main module
===========

How it works
------------

The script will cycle through all .TXT and .xlsx files in a selected directory and fit those files to a selected model. 
You can exclude certain files by using the excludeEnding flag.

The first possible step is to compensate the electrode polarization of the experiment by fitting the data to a cole-cole-model. 
After this fit it will fit the data to a selected model, this can either be the Double Shell model or the Single Shell model.
If there is no electrode polarization to compensate for, you can also skip this. See :py:meth:`impedancefitter.main.Fitter.main`

After the fit to one of the models has finished, the calculated values get written into the 'outfile.yaml' in the data-directory.

The following parameters need to be provided in yaml-files:

Units
-----

Usually, SI units are used.
However, some parameters have a very small numerical value.
In this case, different units are used. 
These parameters are:

+-------------------------+------------------+---------+
| Parameter               | Name in Script   | Unit    |
+=========================+==================+=========+
| :math:`c_0`             | c0               | pF      |
+-------------------------+------------------+---------+
| :math:`c_\mathrm{f}`    | cf               | pF      |
+-------------------------+------------------+---------+
| :math:`C`               | C                | pF      |
+-------------------------+------------------+---------+
| :math:`L`               | L                | nH      |
+-------------------------+------------------+---------+
| :math:`\tau`            | tau              | ps      |
+-------------------------+------------------+---------+


constants.yaml (or alternatively as input dict)
-----------------------------------------------

+-----------------------------------+------------------+-------------------------------------+
| Parameter                         | Name in Script   | Description                         |
+===================================+==================+=====================================+
| :math:`c_0`                       | c0               | air capacitance                     |
+-----------------------------------+------------------+-------------------------------------+
| :math:`c_\mathrm{f}`              | cf               | stray capacitance                   |
+-----------------------------------+------------------+-------------------------------------+
| :math:`p`                         | p                | volume fraction of cells            |
+-----------------------------------+------------------+-------------------------------------+
| :math:`\varepsilon_\mathrm{cp}`   | ecp              | permittivity of cytoplasm           |
+-----------------------------------+------------------+-------------------------------------+
| :math:`d_\mathrm{m}`              | dm               | membrane thickness                  |
+-----------------------------------+------------------+-------------------------------------+
| :math:`R_\mathrm{c}`              | Rc               | outer cell radius                   |
+-----------------------------------+------------------+-------------------------------------+
|                                  only double shell model                                   |
+-----------------------------------+------------------+-------------------------------------+
| :math:`R_\mathrm{n}`              | Rn               | outer Radius of the nucleus         |
+-----------------------------------+------------------+-------------------------------------+
| :math:`d_\mathrm{n}`              | dn               | thickness of the nuclear envelope   |
+-----------------------------------+------------------+-------------------------------------+
| :math:`\varepsilon_\mathrm{np}`   | enp              | permittivity of nucleoplasm         |
+-----------------------------------+------------------+-------------------------------------+

Exemplary file:

.. code-block:: python

    c0 : 3.9e-13    # air capacitance
    cf : 2.42532194241202e-13   # stray capacitance

    Rc : 5.8e-6  # unit(m). cell radius in buffer, this should be changed according to the chosed cell line
    dm : 7.e-9    # thickness of cell membrane m
    ecp : 60     # permittivity for cytoplasm
    p : 0.075

    Rn : 0.8 * constants['Rc']  # radius of nucleus
    dn : 40.e-9   # thickness of nulear membrane
    enp : 120    # Permittivity for nucleoplasm


cole_cole_input.yaml
--------------------

+----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+
| Parameter                        | Name in Script   | Description                                     | Physical Boundaries                                                                 |
+==================================+==================+=================================================+=====================================================================================+
| k                                | k                | constant phase element parameter                | has to be non 0, otherwise the function wil throw NAN or quit(1/k in the formula)   |
+----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+
| :math:`\alpha`                   | alpha            | constant phase element exponent                 | :math:`0<\alpha<1`                                                                  |
+----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+
| :math:`\varepsilon_\mathrm{l}`   | epsi\_l          | low frequency permitivity                       | :math:`el\geq1`                                                                     |
+----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+
| :math:`\varepsilon_\mathrm{h}`   | eh               | high frequency permitivity                      | :math:`eh\geq1`                                                                     |
+----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+
| :math:`\tau`                     | tau              | relaxation time                                 | :math:`\tau>0`                                                                      |
+----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+
| a                                | a                | exponent in formula for :math:`\varepsilon^*`   | :math:`0<a<1`                                                                       |
+----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+
| :math:`\sigma`                   | conductivity     | low frequency conductivity                      | :math:`\sigma > 0`                                                                  |
+----------------------------------+------------------+-------------------------------------------------+-------------------------------------------------------------------------------------+

Exemplary file:

.. code-block:: python

    k: {value: 1.5e-7,  min: 1.e-7, max: 1.e-2, vary: True} 
    epsi_l: {value: 1000, min: 200, max: 3000, vary: True}
    tau: {value: 5.e-7, min: 1.e-8, max: 1.e-3, vary: True}
    a: {value: 0.9, min: 0.8, max: 1.0, vary: True}
    alpha :   {value: 0.3, min: 0., max: .5, vary: True}
    conductivity : {value: 0.05, min: 0.05, max: 0.2, vary: True}
    eh :  {value: 78., min: 60., max: 85., vary: True}

With the `vary` flag, one can choose whether a variable should be included in the fitting procedure or fixed.

single_shell_input.yaml
-----------------------

+-----------------------------------+------------------+-------------------------------------------------------------------------------------------------------------------+------------------------------------------+
| Parameter                         | Name in Script   | Description                                                                                                       | Physical Boundaries                      |
+===================================+==================+===================================================================================================================+==========================================+
| :math:`\sigma_\mathrm{m}`         | km               | conductivity of the membrane                                                                                      | :math:`\sigma_\mathrm{m}>0`              |
+-----------------------------------+------------------+-------------------------------------------------------------------------------------------------------------------+------------------------------------------+
| :math:`\varepsilon_\mathrm{m}`    | em               | pemitivity of the membrane                                                                                        | :math:`\varepsilon_\mathrm{m}\geq1`      |
+-----------------------------------+------------------+-------------------------------------------------------------------------------------------------------------------+------------------------------------------+
| :math:`\sigma_\mathrm{cp}`        | kcp              | conductivity of the cytoplasm                                                                                     | :math:`\sigma_\mathrm{cp} >0`            |
+-----------------------------------+------------------+-------------------------------------------------------------------------------------------------------------------+------------------------------------------+
| :math:`\sigma_\mathrm{med}`       | kmed             | conductivity of the supernatant(\ :math:`\varepsilon_\mathrm{med} = \varepsilon_\mathrm{h}` from cole cole fit)   | :math:`\sigma_\mathrm{med}>0`            |
+-----------------------------------+------------------+-------------------------------------------------------------------------------------------------------------------+------------------------------------------+

Exemplary file:

.. code-block:: python

    km: {value: 1.e-8, min: 1.e-12, max: 1e-2, vary: True} 
    em: {value: 11, min: 1., max: 50., vary: True} 
    kcp: {value: 0.4, min: 0.1, max: 2., vary: True} 
    kmed: {value: 0.15, min: 0.0001, max: 1.0, vary: True} 

With the `vary` flag, one can choose whether a variable should be included in the fitting procedure or fixed.

.. note::

	If you run without electrode polarization, you must include a line for emed as here:
   
    .. code-block:: python

    	emed: {value: 78, min: 70, max: 80, vary: True} 

double_shell_input.yaml
-----------------------

+-----------------------------------+------------------+-------------------------------------------------------------------------------------------------------------------+------------------------------------------+
| Parameter                         | Name in Script   | Description                                                                                                       | Physical Boundaries                      |
+===================================+==================+===================================================================================================================+==========================================+
| :math:`\sigma_\mathrm{m}`         | km               | conductivity of the membrane                                                                                      | :math:`\sigma_\mathrm{m}>0`              |
+-----------------------------------+------------------+-------------------------------------------------------------------------------------------------------------------+------------------------------------------+
| :math:`\varepsilon_\mathrm{m}`    | em               | pemitivity of the membrane                                                                                        | :math:`\varepsilon_\mathrm{m}\geq1`      |
+-----------------------------------+------------------+-------------------------------------------------------------------------------------------------------------------+------------------------------------------+
| :math:`\sigma_\mathrm{cp}`        | kcp              | conductivity of the cytoplasm                                                                                     | :math:`\sigma_\mathrm{cp} >0`            |
+-----------------------------------+------------------+-------------------------------------------------------------------------------------------------------------------+------------------------------------------+
| :math:`\varepsilon_\mathrm{ne}`   | ene              | permitivity of the nulear envelope                                                                                | :math:`\varepsilon_\mathrm{ne} \geq1`    |
+-----------------------------------+------------------+-------------------------------------------------------------------------------------------------------------------+------------------------------------------+
| :math:`\sigma_\mathrm{ne}`        | kne              | conductivity of the nuclear envelope                                                                              | :math:`\sigma_\mathrm{ne} >0`            |
+-----------------------------------+------------------+-------------------------------------------------------------------------------------------------------------------+------------------------------------------+
| :math:`\sigma_\mathrm{np}`        | knp              | conductivity of the nucleoplasm                                                                                   | :math:`\sigma_\mathrm{np}>0`             |
+-----------------------------------+------------------+-------------------------------------------------------------------------------------------------------------------+------------------------------------------+
| :math:`\sigma_\mathrm{med}`       | kmed             | conductivity of the supernatant(\ :math:`\varepsilon_\mathrm{med} = \varepsilon_\mathrm{h}` from cole cole fit)   | :math:`\sigma_\mathrm{med}>0`            |
+-----------------------------------+------------------+-------------------------------------------------------------------------------------------------------------------+------------------------------------------+

Exemplary file:

.. code-block:: python

    km: {value: 300.e-6, min: 1.e-8, max: 5.e-4, vary: True} 
    em: {value: 11., min: 1., max: 20., vary: True} 
    kcp: {value: 0.4, min: 0.1, max: 2., vary: True} 
    ene: {value: 50, min: 1., max: 50., vary: True} 
    kne: {value: 2.e-3, min: 1.e-8, max: 1.e-1, vary: True} 
    knp: {value: .8, min: 0.1, max: 1., vary: True} 
    kmed: {value: 0.05, min: 0.04, max: 0.3, vary: True} 

With the `vary` flag, one can choose whether a variable should be included in the fitting procedure or fixed.

.. note::

	If you run without electrode polarization, you must include a line for emed as here:
   
    .. code-block:: python

    	emed: {value: 78, min: 70, max: 80, vary: True} 

possible values
---------------

In Ermolina, I., Polevaya, Y., & Feldman, Y. (2000). Analysis of dielectric spectra of eukaryotic cells by computer modeling. European Biophysics Journal, 29(2), 141–145. https://doi.org/10.1007/s002490050259,
there have been reported upper/lower limits for certain parameters. They could act as a first guess for the bounds of the optimization method.

+-----------------------------------+---------------+---------------+
| Parameter                         | lower limit   | upper limit   |
+===================================+===============+===============+
| :math:`\varepsilon_\mathrm{m}`    | 1.4           | 16.8          |
+-----------------------------------+---------------+---------------+
| :math:`\sigma_\mathrm{m}`         | 8e-8          | 5.6e-5        |
+-----------------------------------+---------------+---------------+
| :math:`\varepsilon_\mathrm{cp}`   | 60            | 77            |
+-----------------------------------+---------------+---------------+
| :math:`\sigma_\mathrm{cp}`        | 0.033         | 1.1           |
+-----------------------------------+---------------+---------------+
| :math:`\varepsilon_\mathrm{ne}`   | 6.8           | 100           |
+-----------------------------------+---------------+---------------+
| :math:`\sigma_\mathrm{ne}`        | 8.3e-5        | 7e-3          |
+-----------------------------------+---------------+---------------+
| :math:`\varepsilon_\mathrm{np}`   | 32            | 300           |
+-----------------------------------+---------------+---------------+
| :math:`\sigma_\mathrm{np}`        | 0.25          | 2.2           |
+-----------------------------------+---------------+---------------+
| R                                 | 3.5e-6        | 10.5e-6       |
+-----------------------------------+---------------+---------------+
| :math:`R_\mathrm{n}`              | 2.95e-6       | 8.85e-6       |
+-----------------------------------+---------------+---------------+
| d                                 | 3.5e-9        | 10.5e-9       |
+-----------------------------------+---------------+---------------+
| :math:`d_\mathrm{n}`              | 2e-8          | 6e-8          |
+-----------------------------------+---------------+---------------+


.. automodule:: impedancefitter.main
        :members:
        :private-members:
