import os

import impedancefitter
import numpy
import pandas

frequencies = numpy.logspace(2, 7, num=50)
R = 100.0
C = 1e-7

# generate model by user-defined circuit
model = "parallel(R, C)"

lmfit_model = impedancefitter.get_equivalent_circuit_model(model)
Z = lmfit_model.eval(omega=2.0 * numpy.pi * frequencies, R=R, C=C)
df = pandas.DataFrame({"freq": frequencies, "real": Z.real, "imag": Z.imag})
df.to_csv("tmp.csv", index=False)

fitter = impedancefitter.Fitter("CSV")
os.remove("tmp.csv")
# First round of fitting
fitter.linkk_test(limits=[-70, 70])
# Adapting the overfitting parameter
fitter.linkk_test(c=0.7, limits=[-10, 10])
# Further adapting the overfitting parameter
fitter.linkk_test(c=0.5)
