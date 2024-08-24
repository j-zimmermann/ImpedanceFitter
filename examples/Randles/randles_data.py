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
import os
import pandas
from matplotlib import rcParams

rcParams['figure.figsize'] = [15, 10]


# parameters
frequencies = numpy.logspace(0, 8)
Rct = 100.
Rs = 20.
Aw = 300.
C0 = 25e-6


# generate model by user-defined circuit
model = 'R_s + parallel(R_ct + W, C)'
lmfit_model = impedancefitter.get_equivalent_circuit_model(model)
# generate data (without noise)
Z = lmfit_model.eval(omega=2. * numpy.pi * frequencies,
                     ct_R=Rct, s_R=Rs,
                     C=C0, Aw=Aw)
data = {'freq': frequencies, 'real': Z.real,
        'imag': Z.imag}
# write data to csv file
df = pandas.DataFrame(data=data)
df.to_csv('test.csv', index=False)

# initialise fitter with verbose output and show fit result
fitter = impedancefitter.Fitter('CSV', LogLevel='DEBUG', show=True)
os.remove('test.csv')

# define model and initial guess
model = 'Randles'
parameters = {'Rct': {'value': 3. * Rct},
              'Rs': {'value': 0.5 * Rs},
              'C0': {'value': 0.1 * C0},
              'Aw': {'value': 1.2 * Aw}}

# run fit
fitter.run(model, parameters=parameters)

# Inspect initial fit
fitter.plot_initial_best_fit()
