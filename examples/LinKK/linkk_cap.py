import impedancefitter
import numpy
import os
import pandas
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams['figure.figsize'] = [15, 10]


# parameters
lowExp = -5
highExp = 5
decades = numpy.log10(10**highExp / 10**lowExp)
pointsperdecade = int(10. * decades)
frequencies = numpy.logspace(lowExp, highExp, num=pointsperdecade)

Rs1 = 100.
Rs2 = 200.
Cs3 = 0.8e-6
Rs4 = 500.
Aw = 1. / (4e-4 * numpy.sqrt(2))


# generate model by user-defined circuit
model = 'R_s1 + parallel(C_s3, R_s2)  + parallel(R_s4, W_s5)'

lmfit_model = impedancefitter.get_equivalent_circuit_model(model)
Z = lmfit_model.eval(omega=2. * numpy.pi * frequencies,
                     s1_R=Rs1,
                     s2_R=Rs2,
                     s3_C=Cs3,
                     s4_R=Rs4,
                     s5_Aw=Aw)


data = {'freq': frequencies, 'real': Z.real,
        'imag': Z.imag}
# write data to csv file
df = pandas.DataFrame(data=data)
df.to_csv('test.csv', index=False)

fitter = impedancefitter.Fitter('CSV')
os.remove('test.csv')
fitter.visualize_data()

results, mus = fitter.linkk_test(capacitance=True)

RCperdec = numpy.linspace(1.0, len(mus['test.csv0']), num=len(mus['test.csv0'])) / decades
print(RCperdec)
plt.plot(RCperdec, mus['test.csv0'])
plt.show()
