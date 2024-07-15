#    The ImpedanceFitter is a package to fit impedance spectra to
#    equivalent-circuit models using open-source software.
#
#    Copyright (C) 2018, 2019 Leonard Thiele, leonard.thiele[AT]uni-rostock.de
#    Copyright (C) 2018 - 2024 Julius Zimmermann, julius.zimmermann[AT]uni-rostock.de
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

import os

import matplotlib.pyplot as plt
import numpy
import pandas
from matplotlib import rcParams

import impedancefitter

rcParams["figure.figsize"] = [15, 10]


# parameters
lowExp = -5
highExp = 5
decades = numpy.log10(10**highExp / 10**lowExp)
pointsperdecade = int(10.0 * decades)
frequencies = numpy.logspace(lowExp, highExp, num=pointsperdecade)

Rs1 = 100.0
Rs2 = 200.0
Cs3 = 0.8e-6
Rs4 = 500.0
Aw = 1.0 / (4e-4 * numpy.sqrt(2))


# generate model by user-defined circuit
model = "R_s1 + parallel(C_s3, R_s2)  + parallel(R_s4, W_s5)"

lmfit_model = impedancefitter.get_equivalent_circuit_model(model)
Z = lmfit_model.eval(
    omega=2.0 * numpy.pi * frequencies, s1_R=Rs1, s2_R=Rs2, s3_C=Cs3, s4_R=Rs4, s5_Aw=Aw
)


data = {"freq": frequencies, "real": Z.real, "imag": Z.imag}
# write data to csv file
df = pandas.DataFrame(data=data)
df.to_csv("test.csv", index=False)

fitter = impedancefitter.Fitter("CSV")
os.remove("test.csv")

results, mus, residuals = fitter.linkk_test(capacitance=True)

RCperdec = (
    numpy.linspace(1.0, len(mus["test.csv0"]), num=len(mus["test.csv0"])) / decades
)
print(RCperdec)
plt.plot(RCperdec, mus["test.csv0"])
plt.show()
