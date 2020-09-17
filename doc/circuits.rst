.. _circuits:

Circuits
========

There exist a few predefined circuits that were implemented
based on published papers.
Usually, those circuits are rather complex and cannot be built
by the existing elements or feature parameters in certain ranges
or units that are not consistent with the generally chosen unit set.

Names for building the model
----------------------------

To build your equivalent circuit model, choose a model from the following list.
The function, which tells you the parameter names for the respective model is linked
to each model. 

+-----------------------+-----------------------------------------------------------+
| Name                  | Corresponding function                                    |
+=======================+===========================================================+
| ColeCole              | :meth:`impedancefitter.cole_cole.cole_cole_model`         |
+-----------------------+-----------------------------------------------------------+
| ColeCole2             | :meth:`impedancefitter.cole_cole.cole_cole_2_model`       |
+-----------------------+-----------------------------------------------------------+
| ColeCole2tissue       | :meth:`impedancefitter.cole_cole.cole_cole_2tissue_model` |
+-----------------------+-----------------------------------------------------------+
| ColeCole3             | :meth:`impedancefitter.cole_cole.cole_cole_3_model`       |
+-----------------------+-----------------------------------------------------------+
| ColeCole4             | :meth:`impedancefitter.cole_cole.cole_cole_4_model`       |
+-----------------------+-----------------------------------------------------------+
| ColeColeR             | :meth:`impedancefitter.cole_cole.cole_cole_R_model`       |
+-----------------------+-----------------------------------------------------------+
| Randles               | :meth:`impedancefitter.randles.Z_randles`                 |
+-----------------------+-----------------------------------------------------------+
| RandlesCPE            |   :meth:`impedancefitter.randles.Z_randles_CPE`           |
+-----------------------+-----------------------------------------------------------+
| DRC                   |   :meth:`impedancefitter.RC.drc_model`                    |
+-----------------------+-----------------------------------------------------------+
| RCfull                | :meth:`impedancefitter.RC.RC_model`                       |
+-----------------------+-----------------------------------------------------------+
| RC                    | :meth:`impedancefitter.RC.rc_model`                       |
+-----------------------+-----------------------------------------------------------+
| RCtau                 | :meth:`impedancefitter.RC.rc_tau_model`                   |
+-----------------------+-----------------------------------------------------------+
| SingleShell           | :meth:`impedancefitter.single_shell.single_shell_model`   |
+-----------------------+-----------------------------------------------------------+
| DoubleShell           | :meth:`impedancefitter.double_shell.double_shell_model`   |
+-----------------------+-----------------------------------------------------------+
| CPE                   | :meth:`impedancefitter.cpe.cpe_model`                     |
+-----------------------+-----------------------------------------------------------+
| CPECT                 | :meth:`impedancefitter.cpe.cpe_ct_model`                  |
+-----------------------+-----------------------------------------------------------+
| CPECTW                | :meth:`impedancefitter.cpe.cpe_ct_w_model`                |
+-----------------------+-----------------------------------------------------------+
| CPETissue             | :meth:`impedancefitter.cpe.cpetissue_model`               |
+-----------------------+-----------------------------------------------------------+
| CPECTTissue           | :meth:`impedancefitter.cpe.cpe_ct_tissue_model`           |
+-----------------------+-----------------------------------------------------------+
| CPEonset              | :meth:`impedancefitter.cpe.cpe_onset_model`               |
+-----------------------+-----------------------------------------------------------+
| LR                    | :meth:`impedancefitter.loss.Z_in`                         |
+-----------------------+-----------------------------------------------------------+
| LCR                   | :meth:`impedancefitter.loss.Z_loss`                       |
+-----------------------+-----------------------------------------------------------+
| HavriliakNegami       | :meth:`impedancefitter.cole_cole.havriliak_negami`        |
+-----------------------+-----------------------------------------------------------+
| HavriliakNegamiTissue | :meth:`impedancefitter.cole_cole.havriliak_negamitissue`  |
+-----------------------+-----------------------------------------------------------+

Cole-Cole circuits
------------------

.. automodule:: impedancefitter.cole_cole
          :members:


Single-Shell model
------------------

.. automodule:: impedancefitter.single_shell
        :members:

Double-Shell model
------------------

.. automodule:: impedancefitter.double_shell
        :members:

Inductance circuits
-------------------

.. automodule:: impedancefitter.loss
        :members:

CPE circuits
------------

.. automodule:: impedancefitter.cpe
	:members:

RC circuits
-----------

.. automodule:: impedancefitter.RC
	:members:

Randles circuits
----------------

.. automodule:: impedancefitter.randles
	:members:

