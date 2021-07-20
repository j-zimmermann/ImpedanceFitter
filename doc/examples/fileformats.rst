Available file formats
----------------------

We have analysed impedance data measured with
different devices.
Here, we show how the different file formats,
in which the data are stored, can be read in.
Please let us know if you have used ImpedanceFitter
with other file formats.

Gamry Reference 600+
^^^^^^^^^^^^^^^^^^^^

The data are saved in the DTA format.
They can be read in using

.. code-block:: python

       fitter = impedancefitter.Fitter("DTA")

Sciospec ISX-3
^^^^^^^^^^^^^^

The data are stored in `.spec` files, which are plain text files
and can be read in using

.. code-block:: python

       fitter = impedancefitter.Fitter("TXT", ending=".spec", skiprows_txt=6, delimiter=",")

Keysight E4980AL
^^^^^^^^^^^^^^^^

The data were stored through the USB interface and stored
in a CSV file that is structured like:
`frequency, real part of impedance, imaginary part of impedance, voltage, current`.
For these files, we wrote a special function to read it in.
It can be accessed through

.. code-block:: python

       fitter = impedancefitter.Fitter("CSV_E4980AL")

Various instruments with data stored in CSV or Excel files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The structure of CSV and Excel files has to be:
`frequency, real part of impedance, imaginary part of impedance, real part, imaginary part, ...`. 
This means that there can be multiple impedance spectra in one file.
However, all spectra have to be measured at the same frequency points,
which must be stored in the first column of the data set.
Then the data can be read in using:

.. code-block:: python

       fitter = impedancefitter.Fitter("CSV")
       fitter = impedancefitter.Fitter("XSLX")
