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

# initialise fitter with verbose output
fitter = impedancefitter.Fitter('CSV', LogLevel='DEBUG')
os.remove('test.csv')

# define model and initial guess
model = 'Randles'
parameters = {'Rct': {'value': 3. * Rct},
              'Rs': {'value': 0.5 * Rs},
              'C0': {'value': 0.1 * C0},
              'Aw': {'value': 1.2 * Aw}}

# run fit
fitter.run(model, parameters=parameters)
