#    The ImpedanceFitter is a package to fit impedance spectra to
#    equivalent-circuit models using open-source software.
#
#    Copyright (C) 2018, 2019 Leonard Thiele, leonard.thiele[AT]uni-rostock.de
#    Copyright (C) 2018, 2019, 2020 Julius Zimmermann, julius.zimmermann[AT]uni-rostock.de
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import impedancefitter
import numpy

# parameters
frequencies = numpy.logspace(0, 8)
Rct = 100.
Rs = 20.
Aw = 300.
C0 = 25e-6


# generate model by user-defined circuit
model = 'R_s + parallel(R_ct + W, C)'
lmfit_model = impedancefitter.get_equivalent_circuit_model(model)
Z = lmfit_model.eval(omega=2. * numpy.pi * frequencies,
                     ct_R=Rct, s_R=Rs,
                     C=C0, Aw=Aw)
impedancefitter.plot_impedance(2. * numpy.pi * frequencies, Z)

# use pre-implemented circuit
model = 'Randles'
lmfit_model = impedancefitter.get_equivalent_circuit_model(model)
Z2 = lmfit_model.eval(omega=2. * numpy.pi * frequencies,
                      Rct=Rct, Rs=Rs,
                      C0=C0, Aw=Aw)

# check that both are equal
if numpy.all(numpy.isclose(Z, Z2)):
    print("Both formulations are equal.")
