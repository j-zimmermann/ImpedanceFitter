.. _elements:

Circuit elements
================

The following elements are available.
Since prefixes are possible, each element is 
referred to as by a special name.
The elements' parameters are called as in the original
function.
This is the concept of LMFIT.

Names for building the model
----------------------------


+-------+------------------------------------------+
| Name  | Corresponding function                   |
+=======+==========================================+
| R     | :meth:`impedancefitter.elements.Z_R`     |
+-------+------------------------------------------+
| C     | :meth:`impedancefitter.elements.Z_C`     |
+-------+------------------------------------------+
| L     | :meth:`impedancefitter.elements.Z_L`     |
+-------+------------------------------------------+
| W     | :meth:`impedancefitter.elements.Z_w`     |
+-------+------------------------------------------+
| Wo    | :meth:`impedancefitter.elements.Z_wo`    |
+-------+------------------------------------------+
| Ws    | :meth:`impedancefitter.elements.Z_ws`    |
+-------+------------------------------------------+
| Cstray| :meth:`impedancefitter.elements.Z_stray` |
+-------+------------------------------------------+
| ADIbR | :meth:`impedancefitter.elements.Z_ADIb_r`|
+-------+------------------------------------------+
| ADIaR | :meth:`impedancefitter.elements.Z_ADIa_r`|
+-------+------------------------------------------+
| ADIIR | :meth:`impedancefitter.elements.Z_ADII_r`|
+-------+------------------------------------------+
| ADIbA | :meth:`impedancefitter.elements.Z_ADIb_a`|
+-------+------------------------------------------+
| ADIaA | :meth:`impedancefitter.elements.Z_ADIa_a`|
+-------+------------------------------------------+
| ADIIA | :meth:`impedancefitter.elements.Z_ADII_a`|
+-------+------------------------------------------+


API reference
-------------

.. automodule:: impedancefitter.elements
        :members:

