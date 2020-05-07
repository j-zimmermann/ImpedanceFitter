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
