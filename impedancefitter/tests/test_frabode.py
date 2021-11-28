from impedancefitter.FRABode2impedance import bode_to_impedance, bode_csv_to_impedance
import numpy as np

freq = np.logspace(1, 6)
attenuation = np.linspace(1, 10)
phase = np.linspace(-90, 0)


def test_bode_to_impedance():
    omega, impedance = bode_to_impedance(freq, attenuation, phase)
    assert np.all(np.isclose(omega, 2. * np.pi * freq))


def test_RandS_bode_csv_to_impedance():
    omega, impedance = bode_csv_to_impedance("impedancefitter/tests/RohdeSchwarz_47R_100R_shunt.csv", "R&S")
    omega_test = 2. * np.pi * np.logspace(1, 7, num=301)
    assert np.all(np.isclose(omega, omega_test, rtol=1e-2))


def test_MokuGo_bode_csv_to_impedance():
    omega, impedance = bode_csv_to_impedance("impedancefitter/tests/MokuGo_47R_100R_shunt.csv", "MokuGo")
    omega_test = 2. * np.pi * np.logspace(1, np.log10(2e7), num=512)
    assert np.all(np.isclose(omega, omega_test))
